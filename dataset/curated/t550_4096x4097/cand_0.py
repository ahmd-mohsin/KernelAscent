import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 550
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _softmax_bias_kernel(
    X, B, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # round softmax result to bf16 (matches torch.softmax output dtype)
    sm_bf = sm.to(tl.bfloat16)

    b = tl.load(B + cols, mask=mask, other=0.0)
    # bf16 add computed in fp32 then rounded (matches PyTorch opmath)
    out = (sm_bf.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 GEMM with tensor cores
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _softmax_bias_kernel[(m,)](
            y, self.b2, out,
            y.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
