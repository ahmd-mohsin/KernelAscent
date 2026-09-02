import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 175
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _relu_double_softmax_kernel(
    X_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    # Load row, ReLU (exact in any precision), upcast to fp32 like PyTorch's
    # half softmax (accumulate type = float)
    x = tl.load(X_ptr + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))

    # First softmax (fp32 math, output rounded to fp16 like the reference)
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    y = (e1 / s1).to(tl.float16)

    # Second softmax: input is the fp16-rounded result, upcast again
    y32 = tl.where(mask, y.to(tl.float32), float('-inf'))
    m2 = tl.max(y32, 0)
    e2 = tl.exp(y32 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            h = x @ self.W0
            h = torch.relu(h)
            h = torch.softmax(h, dim=-1)
            return torch.softmax(h, dim=-1)

        # GEMM via cuBLAS (tensor cores), same as reference
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _relu_double_softmax_kernel[(rows,)](
            h, out,
            N=N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
