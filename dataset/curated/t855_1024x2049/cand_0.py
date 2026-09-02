import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 855
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (x * rstd).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = xn * w  # fp16 multiply, matches PyTorch half elementwise mul
    hf = h.to(tl.float32)
    # exact gelu computed in fp32 (PyTorch opmath for half), cast back to half
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    tl.store(Y + row * N + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, y, N, 1e-6,
            BLOCK=BLOCK, num_warps=16,
        )
        return y
