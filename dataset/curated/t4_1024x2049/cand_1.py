import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 4
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _fused_softmax_rms_ln_kernel(
    X_ptr, RMSW_ptr, G_ptr, B_ptr, B5_ptr, Out_ptr,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=-float("inf")).to(tl.float32)

    # ---- softmax (fp32 math, round to bf16 like torch.softmax output) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- RMS norm: fp32 mean of squares, normalize, cast bf16, then * weight ----
    ms = tl.sum(y * y, axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (y * rinv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(RMSW_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 internals, bf16 output) ----
    mu = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d = tl.where(mask, y - mu, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (d * inv * g + b).to(tl.bfloat16).to(tl.float32)

    # ---- scale then bias (each op rounds to bf16, matching separate torch ops) ----
    z = (z * 1.4134).to(tl.bfloat16).to(tl.float32)
    b5 = tl.load(B5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (z + b5).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_row + offs, out, mask=mask)


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
        # cuBLAS tensor-core GEMM
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _fused_softmax_rms_ln_kernel[(M_,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, self.b5, out,
            h.stride(0),
            N=N_,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
