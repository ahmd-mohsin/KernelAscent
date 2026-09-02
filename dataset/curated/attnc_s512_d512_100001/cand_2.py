import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100001
S, D, DT = 512, 512, torch.float16

class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build (and cache) a fused QKV weight so the three projections
        # become a single GEMM (better tensor-core utilization on A100).
        Wqkv = getattr(self, "_Wqkv_cache", None)
        if (
            Wqkv is None
            or Wqkv.device != x.device
            or Wqkv.dtype != self.Wq.dtype
            or getattr(self, "_Wqkv_version", None)
            != (self.Wq._version, self.Wk._version, self.Wv._version)
        ):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv_cache = Wqkv
            self._Wqkv_version = (self.Wq._version, self.Wk._version, self.Wv._version)

        d = self.Wq.shape[1]

        # Single fused GEMM for Q, K, V.
        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        squeeze_out = False
        if x.dim() == 2:
            # (S, D) -> (1, 1, S, D) for SDPA (batch, heads, seq, dim)
            q = q.unsqueeze(0).unsqueeze(0)
            k = k.unsqueeze(0).unsqueeze(0)
            v = v.unsqueeze(0).unsqueeze(0)
            squeeze_out = True
        else:
            # treat leading dims as batch with a single head
            q = q.unsqueeze(-3)
            k = k.unsqueeze(-3)
            v = v.unsqueeze(-3)

        # Fused causal attention (flash / memory-efficient kernel):
        # computes softmax(QK^T / sqrt(d) + causal_mask) @ V without
        # materializing the S x S score matrix in global memory.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=1.0 / math.sqrt(d),
        )

        if squeeze_out:
            out = out.squeeze(0).squeeze(0)
        else:
            out = out.squeeze(-3)
        return out
