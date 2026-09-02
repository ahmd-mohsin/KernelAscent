import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 749
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # load row (fp16) -> fp32
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, then multiply by fp16 weight)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    xh = (x * r).to(tl.float16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)  # fp16
    xh = xh * w  # fp16 arithmetic

    # exact GELU: computed in fp32, output cast back to fp16 (matches torch half gelu)
    xf = xh.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    gh = g.to(tl.float16)

    # ReLU + scale in fp16 (matches reference half ops)
    zero = tl.zeros_like(gh)
    gh = tl.maximum(gh, zero)
    scale = tl.full((1,), 1.2045, tl.float16)
    gh = gh * scale

    # LayerNorm: torch computes in fp32 for half inputs, casts output to fp16
    xf2 = tl.where(mask, gh.to(tl.float32), 0.0)
    mean = tl.sum(xf2, axis=0) / N
    diff = tl.where(mask, xf2 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    ln_g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ln_b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xf2 - mean) * inv * ln_g + ln_b

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS GEMM (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            x, self.rms1_w, self.ln5_g, self.ln5_b, y,
            N, x.stride(0), y.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y
