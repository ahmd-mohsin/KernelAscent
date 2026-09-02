import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 59
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _ln_rms_kernel(X, G, B, W, Y,
                   N, stride_x, stride_y,
                   EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * g + b

    # round to bf16 (matches intermediate materialization in reference)
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    # RMSNorm in fp32 on the bf16-rounded values
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rrms = tl.math.rsqrt(ms + EPS_RMS)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (yf * rrms).to(tl.bfloat16) * w  # bf16 multiply, like reference

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_rms_kernel[(M_,)](
            x, self.ln1_g, self.ln1_b, self.rms2_w, y,
            N, x.stride(0), y.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
