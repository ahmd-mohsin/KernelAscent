import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 814
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _double_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 accumulation, like PyTorch fp16 softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x1 = e / s

    # round intermediate to fp16 (matches reference which materializes fp16 tensor)
    x1 = x1.to(tl.float16).to(tl.float32)
    x1 = tl.where(mask, x1, float('-inf'))

    # second softmax
    m2 = tl.max(x1, axis=0)
    e2 = tl.exp(x1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y = e2 / s2

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS tensor-core GEMM
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _double_softmax_kernel[(Mrows,)](
            x, y, N, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
