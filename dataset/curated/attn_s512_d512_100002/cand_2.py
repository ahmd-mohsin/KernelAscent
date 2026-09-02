import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100002
S, D, DT = 512, 512, torch.bfloat16

class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]

        # Lazily build (and cache) a fused QKV weight so the three projections
        # run as ONE GEMM instead of three separate kernel launches.
        Wqkv = getattr(self, "_Wqkv_cache", None)
        if (
            Wqkv is None
            or Wqkv.device != x.device
            or Wqkv.dtype != self.Wq.dtype
        ):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            object.__setattr__(self, "_Wqkv_cache", Wqkv)

        # Single fused projection: [S, D] @ [D, 3D] -> [S, 3D]
        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # Fused scaled-dot-product attention (softmax(q k^T / sqrt(d)) @ v)
        # Default SDPA scale is 1/sqrt(head_dim) == 1/sqrt(d), matching the
        # reference computation exactly. Uses a fused, memory-efficient kernel
        # instead of materializing scores -> softmax -> matmul separately.
        q4 = q.unsqueeze(0).unsqueeze(0)
        k4 = k.unsqueeze(0).unsqueeze(0)
        v4 = v.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q4, k4, v4)
        return out.squeeze(0).squeeze(0)
