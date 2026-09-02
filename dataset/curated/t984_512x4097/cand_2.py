import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 984
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _softmax_bias_kernel(
    X, B3, B4, OUT,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0)
    # scale in fp32, round to bf16 (matches bf16 elementwise scalar mul), back to fp32
    xs = (x.to(tl.float32) * 1.2993).to(tl.bfloat16).to(tl.float32)
    xs = tl.where(mask, xs, float('-inf'))

    row_max = tl.max(xs, axis=0)
    e = tl.exp(xs - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # softmax output rounded to bf16 (as torch.softmax on bf16 tensor)
    sm_b = sm.to(tl.bfloat16).to(tl.float32)

    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    r1 = (sm_b + b3).to(tl.bfloat16).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    r2 = (r1 + b4).to(tl.bfloat16)

    tl.store(OUT + row * stride_om + offs, r2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _softmax_bias_kernel[(m,)](
            y, self.b3, self.b4, out,
            y.stride(0), out.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
