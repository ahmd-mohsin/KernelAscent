import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 6
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _ln_rms_kernel(
    X, G, B, W, Y,
    N, stride_x, stride_y,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm in fp32 (matches ATen bf16 layer_norm which accumulates in fp32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b

    # round to bf16 (output of layer_norm), then upcast for RMSNorm
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)

    z_bf = (yf * rrms).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (z_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS TF32/BF16 tensor-core GEMM
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _ln_rms_kernel[(M_,)](
            x, self.ln1_g, self.ln1_b, self.rms2_w, y,
            N_, x.stride(0), y.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
