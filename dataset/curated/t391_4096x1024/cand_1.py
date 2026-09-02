import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 391
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _scale_layernorm_kernel(
    X, Y, G, B,
    N,
    scale,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * scale

    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self._Wc = None  # cached combined weight (W0 @ W1), computed lazily

    def _get_combined(self):
        if (self._Wc is None
                or self._Wc.device != self.W0.device
                or self._Wc.dtype != self.W0.dtype):
            # Combine the two projections into one (associativity of matmul).
            # fp16 matmul on A100 accumulates in fp32 via tensor cores.
            self._Wc = (self.W0 @ self.W1).contiguous()
        return self._Wc

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            h = x @ self.W0
            h = h @ self.W1
            h = h * 1.114
            return F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)

        Wc = self._get_combined()
        h = torch.matmul(x, Wc)
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _scale_layernorm_kernel[(Mrows,)](
            h, out, self.ln3_g, self.ln3_b,
            N, 1.114, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
