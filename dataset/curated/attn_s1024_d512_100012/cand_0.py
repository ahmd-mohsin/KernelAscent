import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100012
S, D, DT = 1024, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cat_weight(self, x):
        # Cache a single concatenated projection weight so q, k, v are
        # produced with one large GEMM instead of three smaller ones.
        W = getattr(self, "_Wcat", None)
        if (
            W is None
            or W.device != x.device
            or W.dtype != x.dtype
            or W.data_ptr() == 0
        ):
            W = torch.cat(
                (
                    self.Wq.to(device=x.device, dtype=x.dtype),
                    self.Wk.to(device=x.device, dtype=x.dtype),
                    self.Wv.to(device=x.device, dtype=x.dtype),
                ),
                dim=1,
            ).contiguous()
            self._Wcat = W
        return W

    def forward(self, x):
        d = self.Wq.shape[0]

        # Fused QKV projection: one GEMM of (S, D) @ (D, 3D)
        W = self._get_cat_weight(x)
        qkv = x @ W
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        if x.is_cuda:
            # Fused attention (memory-efficient / flash) kernel:
            # avoids materializing the full S x S score matrix in HBM
            # and fuses softmax with both matmuls.
            q4 = q.unsqueeze(0).unsqueeze(0)
            k4 = k.unsqueeze(0).unsqueeze(0)
            v4 = v.unsqueeze(0).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q4, k4, v4,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(d),
            )
            return out.squeeze(0).squeeze(0)
        else:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            a = torch.softmax(scores, dim=-1)
            return a @ v
