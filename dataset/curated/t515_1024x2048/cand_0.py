import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 515
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _softmax_bias_kernel(X, B, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    num = tl.exp(x - x_max)
    den = tl.sum(num, axis=0)
    sm = (num / den).to(tl.float16)

    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float16)
    y = sm + b
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return torch.softmax(x, dim=-1) + self.b1

        x = x.contiguous()
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.view(-1, n_cols)
        n_rows = x2d.shape[0]

        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _softmax_bias_kernel[(n_rows,)](
            x2d, self.b1, y, n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
