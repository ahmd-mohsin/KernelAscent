import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 711
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _softmax_relu_bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to bf16 (match torch.softmax output dtype), relu
    p_bf = p.to(tl.bfloat16)
    zero = tl.zeros_like(p_bf)
    p_bf = tl.maximum(p_bf, zero)

    # add bias in bf16 (match bf16 + bf16 add)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    z_bf = p_bf + b

    # exact (erf-based) gelu in fp32, cast back
    z = z_bf.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(Y_ptr + row * stride_ym + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = x @ self.W0  # (M, 1024) bf16
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_relu_bias_gelu_kernel[(Mrows,)](
            h, self.b3, y,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
