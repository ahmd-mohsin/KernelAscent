import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 858
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_ln_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- load matmul output (bf16) and compute exact GELU in fp32 ----
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round to bf16 (matches F.gelu output dtype), then reload as fp32 for RMS
    g16 = g.to(tl.bfloat16)
    xf = g16.to(tl.float32)

    # ---- RMSNorm ----
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = tl.math.rsqrt(ms + RMS_EPS)
    y16 = (xf * rrms).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y16 = (y16.to(tl.float32) * w).to(tl.bfloat16)

    # ---- LayerNorm (fp32 internal, like PyTorch) ----
    yf = y16.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + LN_EPS)
    gm = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (d * rstd * gm + bt).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_rms_ln_kernel[(m,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, out,
            h.stride(0), out.stride(0),
            n,
            RMS_EPS=1e-6,
            LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
