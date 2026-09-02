import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 554
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, stride_x, stride_y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores on A100)
        z = x @ self.W0
        if not z.is_cuda:
            return torch.relu(torch.softmax(z, dim=-1))
        z = z.contiguous()
        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _softmax_kernel[(m,)](z, out, z.stride(0), out.stride(0), n,
                              BLOCK=BLOCK, num_warps=num_warps)
        # softmax output is nonnegative -> relu is identity
        return out
