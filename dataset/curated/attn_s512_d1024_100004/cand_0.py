import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100004
S, D, DT = 512, 1024, torch.float16

class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None  # lazy fused weight cache

    def forward(self, x):
        # Lazily build fused QKV weight (single wide GEMM instead of 3 GEMMs)
        Wqkv = self._Wqkv
        if (Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype):
            Wqkv = torch.cat((self.Wq, self.Wk, self.Wv), dim=1).to(
                device=x.device, dtype=x.dtype
            ).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv                       # (S, 3D) in one GEMM
        q, k, v = qkv.split(d, dim=-1)

        # Fused attention (softmax(q k^T / sqrt(d)) v) via SDPA kernel
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            scale=1.0 / math.sqrt(d),
        )
        return out.squeeze(0)
