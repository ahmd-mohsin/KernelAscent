import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 114
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _softmax_relu_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0
        if not z.is_cuda:
            return torch.relu(torch.softmax(z, dim=-1))
        z = z.contiguous()
        out = torch.empty_like(z)
        n_rows, n_cols = z.shape
        BLOCK = triton.next_power_of_2(n_cols)
        _softmax_relu_kernel[(n_rows,)](
            z, out, n_cols, z.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
