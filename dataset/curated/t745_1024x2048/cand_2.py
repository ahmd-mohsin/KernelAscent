import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 745
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _gelu_ln_kernel(X, G, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: 0.5*x*(1+erf(x/sqrt(2)))
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g - mean) * rstd * gamma + beta
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _gelu_ln_kernel[(m,)](
            h, self.ln2_g, self.ln2_b, y, n, 1e-5,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
