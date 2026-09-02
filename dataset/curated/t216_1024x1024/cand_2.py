import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 216
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _ln_rms_kernel(
    X, G, B, W, Y,
    N, stride_x, stride_y,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching F.layer_norm on bf16 input)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.math.rsqrt(var + EPS_LN)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (x - mean) * inv * g + b

    # round to bf16 (output of layer_norm), then back to fp32 for RMS stage
    ln_bf16 = ln.to(tl.bfloat16)
    xf = ln_bf16.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    rinv = tl.math.rsqrt(ms + EPS_RMS)
    rms_bf16 = (xf * rinv).to(tl.bfloat16)

    # bf16 multiply with rms2_w (fp32 compute, round to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (rms_bf16.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS tensor-core matmul
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _ln_rms_kernel[(Mrows,)](
            x, self.ln1_g, self.ln1_b, self.rms2_w, y,
            N, x.stride(0), y.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
