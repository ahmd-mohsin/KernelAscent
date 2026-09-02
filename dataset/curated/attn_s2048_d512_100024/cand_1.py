import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100024
S, D, DT = 2048, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None  # lazily built fused projection weight

    def forward(self, x):
        d = x.shape[-1]

        # Build (and cache) fused QKV weight: one big GEMM instead of three.
        if (self._Wqkv is None
                or self._Wqkv.device != x.device
                or self._Wqkv.dtype != x.dtype):
            self._Wqkv = torch.cat(
                (self.Wq, self.Wk, self.Wv), dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()

        qkv = x @ self._Wqkv                     # (S, 3D) single GEMM
        q, k, v = qkv.split(d, dim=-1)

        # Fused attention (memory-efficient / flash kernel) - avoids
        # materializing the (S, S) score matrix in fp32 round trips.
        q4 = q.unsqueeze(0).unsqueeze(0)         # (1, 1, S, D)
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
