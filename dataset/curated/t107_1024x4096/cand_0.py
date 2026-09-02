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
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X_ptr + row * N + offs

    x = tl.load(ptr, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 math, like PyTorch's opmath for bf16)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    y = e1 / s1

    # emulate the cast back to bf16 between the two softmaxes
    y = y.to(tl.bfloat16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))

    # second softmax
    m2 = tl.max(y, axis=0)
    e2 = tl.exp(y - m2)
    s2 = tl.sum(e2, axis=0)
    z = e2 / s2

    # emulate cast to bf16 then scale in fp32 (opmath), cast back
    z = z.to(tl.bfloat16).to(tl.float32) * scale

    tl.store(Y_ptr + row * N + offs, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS/tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _double_softmax_scale_kernel[(rows,)](
            h, out, N, 1.0936,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
