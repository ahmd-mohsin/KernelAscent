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
        d = self.Wq.shape[0]

        # Cache fused QKV weight (one big GEMM instead of three) on the right device.
        wqkv = getattr(self, "_Wqkv", None)
        if wqkv is None or wqkv.device != x.device or wqkv.dtype != x.dtype:
            wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = wqkv

        # Single fused projection: (S, D) @ (D, 3D) -> (S, 3D)
        qkv = x @ wqkv
        q, k, v = qkv.split(d, dim=-1)

        # Shape to (batch=1, heads=1, seq, head_dim) for fused attention kernel.
        q = q.unsqueeze(0).unsqueeze(0)
        k = k.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)

        # Fused causal attention (memory-efficient / flash kernel):
        # equivalent to softmax((q k^T)/sqrt(d) + causal_mask) @ v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return out.squeeze(0).squeeze(0)
