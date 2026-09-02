import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 771
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _gelu_bias_softmax_kernel(
    X, B, Out,
    N,
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # exact GELU in fp32, rounded back to bf16 (matches F.gelu on bf16)
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)

    # bf16 add computed in fp32 opmath, rounded back to bf16 (matches x + b)
    s = (g_bf.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (matches torch.softmax on bf16)
    sf = s.to(tl.float32)
    sf = tl.where(mask, sf, float("-inf"))
    row_max = tl.max(sf, axis=0)
    e = tl.exp(sf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.bfloat16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (bf16, tensor cores on A100)
        y = x @ self.W0
        y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)

        _gelu_bias_softmax_kernel[(Mrows,)](
            y, self.b2, out,
            N,
            y.stride(0),
            out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
