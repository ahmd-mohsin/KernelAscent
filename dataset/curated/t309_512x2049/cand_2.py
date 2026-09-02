import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 309
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _double_softmax_bias_kernel(
    X_ptr, B_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # first softmax (fp32 accumulate, round to fp16 like PyTorch half softmax)
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, 0)
    y1 = (e1 / s1).to(tl.float16).to(tl.float32)
    y1 = tl.where(mask, y1, -float('inf'))

    # second softmax
    m2 = tl.max(y1, 0)
    e2 = tl.exp(y1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    y2 = (e2 / s2).to(tl.float16)

    # bias add in fp16 (matches reference fp16 + fp16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    out = y2 + b
    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 tensor-core GEMM
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _double_softmax_bias_kernel[(rows,)](
            h, self.b3, out, N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
