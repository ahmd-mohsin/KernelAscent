import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 114
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = torch.matmul(x, self.W0)  # cuBLAS tensor-core GEMM
        if not z.is_cuda:
            return torch.relu(torch.softmax(z, dim=-1))
        z = z.contiguous()
        out = torch.empty_like(z)
        Mrows, N = z.shape
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _softmax_kernel[(Mrows,)](
            z, out, N, z.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        # relu(softmax(x)) == softmax(x) since softmax outputs are non-negative
        return out
