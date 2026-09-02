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

    def forward(self, x):
        # Lazily build a fused QKV weight so all three projections run as one GEMM.
        Wqkv = getattr(self, "_Wqkv", None)
        if (Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype):
            Wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv  # single large GEMM: (S, 3D)
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        # Fused attention (flash / memory-efficient kernel): softmax(q k^T / sqrt(d)) v
        # Default SDPA scale is 1/sqrt(last_dim) == 1/sqrt(d), matching the reference.
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
        )
        return out.squeeze(0).squeeze(0)
