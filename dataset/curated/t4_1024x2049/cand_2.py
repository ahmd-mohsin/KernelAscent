import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 4
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, RW_ptr, G_ptr, B_ptr, B5_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- softmax (fp32 math, round to bf16 like torch.softmax) ----
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask,
                other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # ---- RMS norm (fp32 from bf16, round to bf16, then * w in fp32, round) ----
    xf = sm.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = (xf * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16)
    w = tl.load(RW_ptr + offs, mask=mask, other=0).to(tl.float32)
    h = (r.to(tl.float32) * w).to(tl.bfloat16)

    # ---- LayerNorm (fp32 accumulation, affine in fp32, round to bf16) ----
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    mean = tl.sum(hf, axis=0) / N
    d = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = ((d * inv) * g + b).to(tl.bfloat16)

    # ---- scale (fp32 mul, round to bf16) ----
    y = (y.to(tl.float32) * 1.4134).to(tl.bfloat16)

    # ---- bias add (fp32 add, round to bf16) ----
    b5 = tl.load(B5_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = (y.to(tl.float32) + b5).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0  # (M, 512), bf16
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, self.b5, y,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
