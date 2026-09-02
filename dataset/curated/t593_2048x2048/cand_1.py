import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 593
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b2_ptr, w_ptr, out_ptr, N, stride_row,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu 1 (exact, computed in fp32, cast back to fp16 to match PyTorch half kernels)
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # gelu 2
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # + b2 (half add, opmath fp32, cast back)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b2)
    x = x.to(tl.float16).to(tl.float32)

    # rmsnorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    y = (x * inv).to(tl.float16).to(tl.float32)

    # * rms3_w (half mul, opmath fp32, cast back)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.float16).to(tl.float32)

    # relu
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.b2, self.rms3_w, out, N, x.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
