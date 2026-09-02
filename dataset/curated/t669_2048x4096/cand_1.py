import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 669
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_ln_rms_relu_rms(
    X, OUT, G1, B1, W2, W4,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, bf16 output like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * inv * g1 + b1).to(tl.bfloat16)

    # ---- RMSNorm #1 (compute on bf16 values cast to fp32) ----
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    z = ((yf * r).to(tl.bfloat16) * w2)  # bf16 multiply

    # ---- ReLU (bf16) ----
    zero = tl.zeros_like(z)
    z = tl.maximum(z, zero)

    # ---- RMSNorm #2 ----
    zf = z.to(tl.float32)
    ms2 = tl.sum(zf * zf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + EPS_RMS)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    out = ((zf * r2).to(tl.bfloat16) * w4)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_rms_relu_rms[(Mrows,)](
            h, out, self.ln1_g, self.ln1_b, self.rms2_w, self.rms4_w,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            EPS_LN=1e-5, EPS_RMS=1e-6,
            num_warps=8,
        )
        return out
