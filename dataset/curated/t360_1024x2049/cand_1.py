import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 360
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b1_ptr, w_ptr, out_ptr, N, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)

    # x = x * 1.0393 + b1 (in fp16 to match reference numerics)
    scale = tl.full((), 1.0393, tl.float32).to(x.dtype)
    x = x * scale + b1

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)

    y = (xf * rstd).to(x.dtype)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = y * w

    tl.store(out_ptr + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x, self.b1, self.rms2_w, out,
            N, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
