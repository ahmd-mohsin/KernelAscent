import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 107
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _double_softmax_scale_kernel(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)

    # First softmax (fp32 accumulation, like PyTorch's bf16 softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1

    # Round to bf16 to match the intermediate tensor dtype in the reference
    p1 = p1.to(tl.bfloat16).to(tl.float32)
    p1 = tl.where(mask, p1, float('-inf'))

    # Second softmax
    m2 = tl.max(p1, axis=0)
    e2 = tl.exp(p1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    # Scale in fp32 (opmath), then cast to bf16
    out = p2 * scale
    tl.store(Y_ptr + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (bf16, same as reference)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _double_softmax_scale_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            n, 1.0936,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
