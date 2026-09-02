import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 564
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _relu_bias_softmax_kernel(
    Y_ptr, B_ptr, OUT_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    y = tl.load(Y_ptr + row * stride_row + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # relu in fp16, add bias in fp16 (matches reference rounding)
    z_h = tl.maximum(y, 0.0) + b
    z = z_h.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))

    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(OUT_ptr + row * stride_row + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # First GEMM (cuBLAS tensor cores)
        h = x @ self.W0
        # In-place scale (same rounding as reference elementwise mul)
        h.mul_(1.4351)
        # Second GEMM
        y = h @ self.W2

        if not y.is_cuda:
            z = torch.relu(y) + self.b4
            return torch.softmax(z, dim=-1)

        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _relu_bias_softmax_kernel[(rows,)](
            y, self.b4, out,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
