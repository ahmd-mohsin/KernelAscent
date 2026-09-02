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
    N, stride_x, stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)

    # softmax (fp32 accumulate, matching PyTorch's internal upcast)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    sm = e / s

    # round to bf16 (softmax output dtype), relu is identity on nonneg values
    sm_bf = sm.to(tl.bfloat16)
    sm_bf = tl.maximum(sm_bf, 0.0)

    # add bias: bf16 inputs, fp32 compute, round to bf16
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    a = (sm_bf.to(tl.float32) + b).to(tl.bfloat16)

    # gelu (erf-based), fp32 compute, round to bf16
    af = a.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * af * (1.0 + tl.math.erf(af * INV_SQRT2))

    tl.store(Y_ptr + row * stride_y + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _softmax_relu_bias_gelu_kernel[(Mrows,)](
            h, self.b3, y,
            N, h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
