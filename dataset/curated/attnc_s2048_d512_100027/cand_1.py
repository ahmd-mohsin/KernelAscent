import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100027
S, D, DT = 2048, 512, torch.bfloat16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]

        # Lazily build a fused QKV weight so all three projections run as one GEMM.
        Wqkv = getattr(self, "_Wqkv_cache", None)
        if (
            Wqkv is None
            or Wqkv.device != self.Wq.device
            or Wqkv.dtype != self.Wq.dtype
        ):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv_cache = Wqkv

        # Single fused projection: (S, D) @ (D, 3D) -> (S, 3D)
        qkv = x @ Wqkv
        q, k, v = qkv.chunk(3, dim=-1)

        s = x.shape[0]
        # Shape to (batch=1, heads=1, seq, dim) for fused SDPA.
        q = q.reshape(1, 1, s, d)
        k = k.reshape(1, 1, s, d)
        v = v.reshape(1, 1, s, d)

        # Fused causal attention (flash / memory-efficient kernel):
        # computes softmax((q k^T)/sqrt(d) + causal_mask) v with fp32 accumulation,
        # matching the reference math without materializing the S x S score matrix.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return out.reshape(s, d)
