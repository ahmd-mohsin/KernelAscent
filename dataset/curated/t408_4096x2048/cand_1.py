import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 408
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _double_softmax_kernel(
    X_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # first softmax (fp32 accumulation, like PyTorch's half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    p = e1 / s1

    # emulate the fp16 round-trip of the intermediate tensor
    p = p.to(tl.float16).to(tl.float32)
    p = tl.where(mask, p, -float('inf'))

    # second softmax
    m2 = tl.max(p, axis=0)
    e2 = tl.exp(p - m2)
    s2 = tl.sum(e2, axis=0)
    q = e2 / s2

    tl.store(Y_ptr + row * stride_y + offs, q.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        if not h.is_cuda:
            h = torch.softmax(h, dim=-1)
            h = torch.softmax(h, dim=-1)
            return h

        h = h.contiguous()
        rows, n = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _double_softmax_kernel[(rows,)](
            h, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
