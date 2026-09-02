import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 470
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _bias_softmax_kernel(
    X, B, Y,
    n_cols,
    stride_xm, stride_ym,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # bf16 add (match reference rounding), then softmax in fp32
    t = (x + b).to(tl.bfloat16).to(tl.float32)
    t = tl.where(mask, t, float('-inf'))

    row_max = tl.max(t, axis=0)
    num = tl.exp(t - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return torch.softmax(x + self.b0, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)

        BLOCK_SIZE = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4

        _bias_softmax_kernel[(m,)](
            x2, self.b0, y,
            n,
            x2.stride(0), y.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
