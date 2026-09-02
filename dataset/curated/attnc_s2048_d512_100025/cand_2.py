import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100025
S, D, DT = 2048, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build (and cache) a fused QKV weight so the three GEMMs
        # collapse into a single, larger, tensor-core-friendly GEMM.
        Wqkv = getattr(self, "_Wqkv_cache", None)
        if (
            Wqkv is None
            or Wqkv.device != x.device
            or Wqkv.dtype != self.Wq.dtype
        ):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv_cache = Wqkv

        d = self.Wq.shape[0]

        # One fused GEMM for Q, K, V.
        qkv = x @ Wqkv  # (S, 3*D)
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Fused causal attention (memory-efficient / flash kernel):
        # softmax((q k^T)/sqrt(d) + causal_mask) @ v  computed without
        # materializing the (S, S) score matrix; fp32 accumulation inside
        # the kernel matches the reference's fp32 softmax numerics.
        q = q.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
        k = k.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return out.squeeze(0).squeeze(0)
