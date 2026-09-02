import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100006
S, D, DT = 512, 1024, torch.bfloat16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None  # lazily-built fused projection weight

    def _get_fused_weight(self, device, dtype):
        if (
            self._Wqkv is None
            or self._Wqkv.device != device
            or self._Wqkv.dtype != dtype
        ):
            # Concatenate along the output dim so one GEMM produces q|k|v.
            self._Wqkv = torch.cat(
                [self.Wq.to(device=device, dtype=dtype),
                 self.Wk.to(device=device, dtype=dtype),
                 self.Wv.to(device=device, dtype=dtype)],
                dim=1,
            ).contiguous()
        return self._Wqkv

    def forward(self, x):
        d = self.Wq.shape[0]
        W = self._get_fused_weight(x.device, x.dtype)

        # Single fused GEMM for all three projections.
        qkv = x @ W
        q, k, v = qkv.split(d, dim=-1)

        # Scaled attention scores (scale = 1/32 is an exact power of two,
        # so multiplying is bitwise-identical to dividing by sqrt(1024)).
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        # Softmax (CUDA kernel accumulates in fp32 internally, matching reference).
        a = torch.softmax(scores, dim=-1)

        return torch.matmul(a, v)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
