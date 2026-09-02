import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 876
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _softmax_gelu_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch's half softmax)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # round to fp16 (softmax output dtype in reference), then gelu in fp32
    sm_h = sm.to(tl.float16)
    v = sm_h.to(tl.float32)

    # exact GELU: 0.5 * v * (1 + erf(v / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * v * (1.0 + tl.math.erf(v * INV_SQRT2))

    tl.store(Y + row * stride_ym + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _softmax_gelu_kernel[(m,)](
            h, out, n,
            h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
