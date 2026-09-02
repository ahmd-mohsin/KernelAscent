import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 333
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    D: tl.constexpr,
    eps,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load bf16 row, multiply by scalar (f32 math, round to bf16 like PyTorch)
    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    # mean of squares in float32
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + eps)

    # normalize in f32, round to bf16 (matches .to(x.dtype))
    xn = (x * r).to(tl.bfloat16).to(tl.float32)

    # multiply by weight (bf16*bf16 -> single rounding, exact match)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores) — same as reference
        x = torch.matmul(x, self.W0)
        x = x.contiguous()

        m, d = x.shape
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(d)
        _scale_rmsnorm_kernel[(m,)](
            x, self.rms2_w, y,
            D=d,
            eps=1e-6,
            scale=1.4797,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
