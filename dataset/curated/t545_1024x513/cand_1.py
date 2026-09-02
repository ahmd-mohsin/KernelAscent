import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 545
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_ln_softmax_rms_kernel(
    X_ptr, OUT_ptr,
    G_ptr, B_ptr, B3_ptr, RW_ptr,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, round to fp16 like F.layer_norm on half)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation, round to fp16)
    y = tl.where(mask, y, float('-inf'))
    ymax = tl.max(y, axis=0)
    e = tl.exp(y - ymax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # add b3 (fp16 rounding)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (sm + b3).to(tl.float16).to(tl.float32)

    # scalar mul (fp32 math, round to fp16)
    z = (z * SCALE).to(tl.float16).to(tl.float32)

    # RMSNorm (fp32) then * rms5_w in fp16
    z2 = tl.where(mask, z * z, 0.0)
    ms = tl.sum(z2, axis=0) / N
    r = z * (1.0 / tl.sqrt(ms + RMS_EPS))
    rw = tl.load(RW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (r.to(tl.float16).to(tl.float32) * rw).to(tl.float16)

    tl.store(OUT_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax_rms_kernel[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.b3, self.rms5_w,
            N, h.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6, SCALE=1.4014,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
