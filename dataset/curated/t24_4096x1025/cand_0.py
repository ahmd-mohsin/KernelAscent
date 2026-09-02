import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 24
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0)
    # replicate fp16 multiply: x = x * 1.4032 (rounded to fp16)
    xh = (x.to(tl.float32) * SCALE).to(tl.float16)
    xf = xh.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + EPS)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = (xf * rs).to(tl.float16) * w
    tl.store(Y_ptr + row * stride + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.512 if False else torch.randn)(512, 512, generator=g).div_(math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (fp16 tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape

        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _scale_rmsnorm_kernel[(rows,)](
            x, self.rms2_w, y,
            N, x.stride(0),
            SCALE=1.4032, EPS=1e-6, BLOCK=BLOCK,
            num_warps=4,
        )
        return y @ self.W3
