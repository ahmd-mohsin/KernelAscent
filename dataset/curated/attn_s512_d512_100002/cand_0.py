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
        # Fuse the three projections into a single GEMM (cache concatenated weight)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype)
            self._Wqkv = Wqkv

        qkv = x @ Wqkv
        d = x.shape[-1]
        q = qkv[:, :d].unsqueeze(0)
        k = qkv[:, d:2 * d].unsqueeze(0)
        v = qkv[:, 2 * d:].unsqueeze(0)

        # Fused flash-attention kernel (softmax(QK^T/sqrt(d)) @ V) on A100
        out = F.scaled_dot_product_attention(q, k, v)
        return out.squeeze(0)
