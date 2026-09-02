import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 873
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _gelu_softmax_scale_kernel(
    X_ptr, Y_ptr,
    N,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 like PyTorch opmath
    SQRT1_2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * SQRT1_2))
    # round to bf16 to match PyTorch producing a bf16 intermediate tensor
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (PyTorch upcasts bf16 softmax internally)
    g_masked = tl.where(mask, g, float('-inf'))
    m = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    # round softmax output to bf16 (PyTorch returns bf16 tensor)
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # scale by 1.1536 in fp32 opmath, output bf16 (relu is identity: values >= 0)
    out = sm * SCALE
    tl.store(Y_ptr + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _gelu_softmax_scale_kernel[(Mrows,)](
            h, y,
            N,
            h.stride(0), y.stride(0),
            SCALE=1.1536,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
