import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 119
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride,
    eps,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul output (bf16)
    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0)

    # emulate: x = x * 1.4953 in bf16 (round to bf16), then upcast to fp32
    xf = (x.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)

    # RMS in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    # normalized value cast to bf16, then multiply by bf16 weight
    xn = (xf * r).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = xn * w

    tl.store(Y_ptr + row * stride + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _scale_rmsnorm_kernel[(Mrows,)](
            y, self.rms2_w, out,
            N, y.stride(0),
            1e-6,
            1.4953,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
