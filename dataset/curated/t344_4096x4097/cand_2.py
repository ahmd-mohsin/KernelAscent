import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 344
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_scale_bias_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, computed in fp32 then rounded to bf16
    # to mirror PyTorch's per-op bf16 rounding
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    g = (g * 1.0846).to(tl.bfloat16).to(tl.float32)
    g = (g * 1.379).to(tl.bfloat16).to(tl.float32)

    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax over the row (fp32 accumulation, like PyTorch's bf16 softmax)
    g_m = tl.where(mask, g, float("-inf"))
    row_max = tl.max(g_m, axis=0)
    e = tl.exp(g_m - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (bf16, tensor cores on A100)
        y = x @ self.W0
        y = y.contiguous()

        M_, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_scale_bias_softmax[(M_,)](
            y, self.b4, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
