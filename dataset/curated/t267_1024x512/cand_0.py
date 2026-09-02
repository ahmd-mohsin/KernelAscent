import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 267
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _gelu_bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # GELU (exact, erf-based) computed in fp32, rounded back to bf16
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)

    # bias add: fp32 opmath then round to bf16 (matches PyTorch bf16 add)
    s = (g_bf.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32
    sf = tl.where(mask, s.to(tl.float32), float('-inf'))
    m = tl.max(sf, axis=0)
    e = tl.exp(sf - m)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _gelu_bias_softmax_kernel[(Mrows,)](
            y, self.b2, out,
            N, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
