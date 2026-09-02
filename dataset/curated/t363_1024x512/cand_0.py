import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 363
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, stride_x, stride_y, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)

    # normalized, rounded to bf16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> computed in fp32, rounded back to bf16 (PyTorch opmath semantics)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # exact GELU in fp32, cast to bf16
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, y, N,
            x.stride(0), y.stride(0),
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
