import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 184
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_softmax_scale_ln_kernel(
    X_ptr, G_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- load matmul output row, upcast to fp32 (matches PyTorch bf16 softmax) ----
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax in fp32 ----
    m = tl.max(x, axis=0)
    e = tl.math.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to bf16 like the eager softmax output, then re-upcast
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- scale (opmath fp32, result rounded to bf16 as in eager) ----
    p = p * 1.2551
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- relu ----
    p = tl.maximum(p, 0.0)

    # ---- layernorm in fp32 (matches PyTorch bf16 layer_norm internals) ----
    p_masked = tl.where(mask, p, 0.0)
    mean = tl.sum(p_masked, axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores, fp32 accumulate — same as eager)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        grid = (rows,)
        _fused_softmax_scale_ln_kernel[grid](
            h, self.ln4_g, self.ln4_b, out,
            N, h.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
