import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100014
S, D, DT = 1024, 512, torch.bfloat16

class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build a fused QKV weight (single wide GEMM instead of three)
        Wqkv = getattr(self, "_Wqkv", None)
        if (
            Wqkv is None
            or Wqkv.device != x.device
            or Wqkv.dtype != self.Wq.dtype
        ):
            Wqkv = torch.cat((self.Wq, self.Wk, self.Wv), dim=1).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[1]
        qkv = x @ Wqkv  # (S, 3D) in one GEMM
        q, k, v = qkv.split(d, dim=-1)

        # Fused flash-attention kernel (scale = 1/sqrt(d) is the default)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        )
        return out.squeeze(0)
