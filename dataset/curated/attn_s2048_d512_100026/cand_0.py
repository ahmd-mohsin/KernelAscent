import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100026
S, D, DT = 2048, 512, torch.bfloat16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_weight(self, device, dtype):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != device or w.dtype != dtype:
            w = torch.cat(
                [
                    self.Wq.to(device=device, dtype=dtype),
                    self.Wk.to(device=device, dtype=dtype),
                    self.Wv.to(device=device, dtype=dtype),
                ],
                dim=1,
            ).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = x.shape[-1]
        Wqkv = self._get_fused_weight(x.device, x.dtype)

        # Single fused GEMM for Q, K, V projections
        qkv = x @ Wqkv
        q, k, v = qkv.chunk(3, dim=-1)

        if x.is_cuda:
            # Fused (memory-efficient / flash) attention: avoids materializing
            # the full S x S score matrix in global memory and fuses
            # scale + softmax + PV matmul. fp32 accumulation internally.
            q4 = q.unsqueeze(0).unsqueeze(0)
            k4 = k.unsqueeze(0).unsqueeze(0)
            v4 = v.unsqueeze(0).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q4, k4, v4, scale=1.0 / math.sqrt(d)
            )
            return out.squeeze(0).squeeze(0)
        else:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            a = torch.softmax(scores, dim=-1)
            return a @ v
