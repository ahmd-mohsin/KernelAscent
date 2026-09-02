import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 793
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, RMSW_ptr, G4_ptr, B4_ptr, G5_ptr, B5_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)

    # ---- softmax (fp32 accum, output rounded to fp16 like PyTorch) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # ---- gelu (erf, fp32 opmath, rounded to fp16) ----
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # ---- RMSNorm (explicit fp32, cast fp16, then fp16 multiply by weight) ----
    ms = tl.sum(g * g, axis=0) / N
    r = (g * tl.math.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)
    w = tl.load(RMSW_ptr + offs).to(tl.float32)
    v = (r * w).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 1 (fp32 opmath, output fp16) ----
    g4 = tl.load(G4_ptr + offs).to(tl.float32)
    b4 = tl.load(B4_ptr + offs).to(tl.float32)
    mean1 = tl.sum(v, axis=0) / N
    d1 = v - mean1
    var1 = tl.sum(d1 * d1, axis=0) / N
    y1 = d1 * tl.math.rsqrt(var1 + 1e-5) * g4 + b4
    y1 = y1.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 (fp32 opmath, output fp16) ----
    g5 = tl.load(G5_ptr + offs).to(tl.float32)
    b5 = tl.load(B5_ptr + offs).to(tl.float32)
    mean2 = tl.sum(y1, axis=0) / N
    d2 = y1 - mean2
    var2 = tl.sum(d2 * d2, axis=0) / N
    y2 = d2 * tl.math.rsqrt(var2 + 1e-5) * g5 + b5

    tl.store(Y_ptr + row * stride_y + offs, y2.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        grid = (Mrows,)
        _fused_post_kernel[grid](
            h, self.rms3_w, self.ln4_g, self.ln4_b, self.ln5_g, self.ln5_b, y,
            h.stride(0), y.stride(0),
            N=N, BLOCK=4096,
            num_warps=8,
        )
        return y
