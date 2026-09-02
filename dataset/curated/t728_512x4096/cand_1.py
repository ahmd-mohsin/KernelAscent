import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 728
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _softmax_relu_kernel(
    X, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float("-inf")).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    # relu of softmax output is identity (non-negative), kept implicitly
    tl.store(Y + row * stride_y + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (already optimal on A100 tensor cores)
        h = torch.matmul(x, self.W0)
        if not h.is_cuda:
            return torch.relu(torch.softmax(h, dim=-1))
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_relu_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
