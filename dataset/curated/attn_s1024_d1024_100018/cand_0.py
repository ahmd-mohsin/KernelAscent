import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100018
S, D, DT = 1024, 1024, torch.bfloat16

class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fuse the three projections into a single GEMM (cached, moved to x's device once)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        qkv = x @ Wqkv  # (S, 3D) in one tensor-core GEMM
        d = self.Wq.shape[1]
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Fused flash-attention kernel (softmax(QK^T/sqrt(d)) @ V) — no S x S
        # materialization, single fused kernel on A100.
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            scale=1.0 / math.sqrt(d),
        )
        return out.squeeze(0)
