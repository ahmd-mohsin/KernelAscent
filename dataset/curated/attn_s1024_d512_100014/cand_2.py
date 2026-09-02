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
        # Lazily build a fused QKV weight so all three projections run as
        # a single large GEMM (better tensor-core utilization on A100).
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv  # (S, 3D) single GEMM
        q, k, v = qkv.split(d, dim=-1)

        # Fused, memory-efficient attention kernel (avoids materializing the
        # full S x S score matrix in global memory + fuses softmax).
        # Default scale = 1/sqrt(E) matches the reference exactly.
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
        )
        return out.squeeze(0).squeeze(0)
