import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 816
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _scale_gelu_softmax_kernel(
    X_ptr, Out_ptr,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    # scale in fp32, round to bf16 (matches x = x * 1.482 on bf16 tensor)
    x = (x.to(tl.float32) * 1.482).to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU, computed in fp32, rounded to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax internal float accumulation)
    g_masked = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _scale_gelu_softmax_kernel[(Mrows,)](
            y, out,
            y.stride(0), out.stride(0),
            N, BLOCK,
            num_warps=num_warps,
        )
        return out
