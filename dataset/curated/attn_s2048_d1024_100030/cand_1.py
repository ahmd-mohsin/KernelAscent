import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100030
S, D, DT = 2048, 1024, torch.bfloat16

class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    @torch.no_grad()
    def _build_fused_weight(self, device):
        d = self.Wq.shape[0]
        scale = 1.0 / math.sqrt(d)
        # Fold the softmax scale into Wq only when it is an exact power of two,
        # which guarantees bitwise-identical results (pure exponent shift).
        log2s = math.log2(1.0 / scale)
        exact_pow2 = (log2s == round(log2s))
        Wq = self.Wq
        if exact_pow2:
            Wq = self.Wq * scale  # exact in floating point (power-of-two scale)
        # Single fused QKV weight -> one GEMM instead of three
        Wqkv = torch.cat([Wq, self.Wk, self.Wv], dim=1).to(device).contiguous()
        self._Wqkv = Wqkv
        self._scale_folded = exact_pow2
        self._fused_device = device
        self._d = d

    def forward(self, x):
        if (getattr(self, "_Wqkv", None) is None
                or getattr(self, "_fused_device", None) != x.device):
            self._build_fused_weight(x.device)

        d = self._d
        # One fused GEMM for Q, K, V projections
        qkv = x @ self._Wqkv
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        if self._scale_folded:
            # scale already folded into q (exact power-of-two folding)
            scores = q @ k.transpose(-1, -2)
        else:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)

        a = torch.softmax(scores, dim=-1)
        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
