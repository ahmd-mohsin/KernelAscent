import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 268
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _rmsnorm_scale_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,                      # row length
    stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # mean of squares over the row
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    # normalized, rounded to bf16 (matches .to(x.dtype))
    xn = (x * rstd).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # bf16

    # bf16 * bf16 elementwise -> computed in fp32, rounded to bf16
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    # multiply by scalar in fp32 opmath, round to bf16
    y = (y.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _rmsnorm_scale_kernel[(rows,)](
            x2, self.rms0_w, y,
            N,
            x2.stride(0), y.stride(0),
            1e-6, 1.4423,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
