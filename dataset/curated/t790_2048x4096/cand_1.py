import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 790
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _bias3_softmax_kernel(
    X, B1, B2, B3, Out,
    N, stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)

    # Replicate PyTorch bf16 elementwise adds: fp32 math, round to bf16 each step
    x = (x + b1).to(tl.bfloat16).to(tl.float32)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)
    x = (x + b3).to(tl.bfloat16).to(tl.float32)

    x_m = tl.where(mask, x, float("-inf"))
    row_max = tl.max(x_m, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Out + row * stride_om + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (same as reference matmul)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _bias3_softmax_kernel[(m,)](
            h, self.b1, self.b2, self.b3, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
