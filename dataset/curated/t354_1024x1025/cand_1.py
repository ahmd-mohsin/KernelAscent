import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 354
M, D, DT = 1024, 1025, torch.float16


@triton.jit
def _ln_gelu_relu_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # quantize to fp16 to match layer_norm's fp16 output before gelu
    y = y.to(tl.float16).to(tl.float32)

    # exact (erf-based) GELU in fp32, as PyTorch upcasts half inputs
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = y * 0.5 * (1.0 + tl.math.erf(y * INV_SQRT2))

    # cast to fp16 then relu
    y16 = y.to(tl.float16)
    zero = tl.zeros_like(y16)
    y16 = tl.maximum(y16, zero)

    tl.store(Y + row * stride_y + cols, y16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_gelu_relu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK, EPS=1e-5,
            num_warps=8,
        )
        return out
