import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 592
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X, W_RMS, G, B, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, matching reference)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * r).to(tl.float16)

    # elementwise multiply by rms weight (fp32 compute, fp16 result - matches PyTorch)
    w = tl.load(W_RMS + cols, mask=mask, other=0.0).to(tl.float32)
    z16 = (y16.to(tl.float32) * w).to(tl.float16)

    # LayerNorm (fp32 accumulation, matching PyTorch half layer_norm)
    zf = z16.to(tl.float32)
    mean = tl.sum(zf, axis=0) / N
    zc = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(zc * zc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((zf - mean) * rstd * g + b).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM with fp32 accumulate
        M_, N_ = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _fused_rms_ln_kernel[(M_,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, out,
            x.stride(0), out.stride(0),
            N=N_, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
