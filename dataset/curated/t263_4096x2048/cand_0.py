import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 263
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_bias_softmax_ln_relu(
    X_ptr, Bias_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # load row (bf16) and bias (bf16); add in bf16 to match reference rounding
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(Bias_ptr + cols, mask=mask, other=0.0)
    xb = (x + b).to(tl.float32)
    xb = tl.where(mask, xb, float('-inf'))

    # softmax in fp32 (matches torch's fp32 accumulation), round to bf16
    row_max = tl.max(xb, axis=0)
    e = tl.exp(xb - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.bfloat16).to(tl.float32)
    p = tl.where(mask, p, 0.0)

    # layernorm in fp32 (matches torch's internal fp32 stats)
    mean = tl.sum(p, axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + beta

    # relu, then cast to bf16 (relu commutes with the cast)
    y = tl.maximum(y, 0.0).to(tl.bfloat16)
    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (already optimal on A100)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)

        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _fused_bias_softmax_ln_relu[(m,)](
            h, self.b1, self.ln3_g, self.ln3_b, out,
            n, h.stride(0), out.stride(0), 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
