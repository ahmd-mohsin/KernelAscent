import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100029
S, D, DT = 2048, 1024, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build (and cache) a fused QKV weight so the three projections
        # become a single large GEMM (better tensor-core utilization on A100).
        Wqkv = getattr(self, "_Wqkv", None)
        if (
            Wqkv is None
            or Wqkv.device != self.Wq.device
            or Wqkv.dtype != self.Wq.dtype
        ):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv  # (S, 3D) in one GEMM
        q, k, v = qkv.split(d, dim=-1)

        # Shape to (batch=1, heads=1, seq, dim) for fused SDPA.
        q = q.unsqueeze(0).unsqueeze(0)
        k = k.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)

        # Fused causal attention (memory-efficient/flash kernel):
        #  - avoids materializing the S x S score matrix and the -inf mask
        #  - scale = 1/sqrt(head_dim) matches the reference exactly
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=1.0 / math.sqrt(d),
        )
        return out.squeeze(0).squeeze(0)
