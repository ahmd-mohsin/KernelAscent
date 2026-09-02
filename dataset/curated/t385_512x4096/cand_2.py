import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 385
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _rms_relu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, eps,
    stride_x, stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    # normalize in fp32, round to bf16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    # bf16 * bf16 elementwise in PyTorch computes in fp32, stores bf16
    y = xn.to(tl.float32) * w.to(tl.float32)
    y = tl.maximum(y, 0.0)
    y = y.to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _rms_relu_kernel[(Mrows,)](
            x2, self.rms0_w, y,
            N, 1e-6,
            x2.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
