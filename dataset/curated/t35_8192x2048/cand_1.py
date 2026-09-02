import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 35
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln_rms_relu_kernel(
    X_ptr, G_ptr, B_ptr, W2_ptr, Out_ptr,
    N,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch's opmath for bf16 inputs)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # layer_norm output is cast back to bf16 in the reference
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm branch: _xf = y.float(); rsqrt(mean(y^2)+1e-6)
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    rms = 1.0 / tl.sqrt(ms + EPS_RMS)

    # (_xf * rms).to(bf16) * rms2_w  (bf16 elementwise mul == fp32 mul + round)
    a = (y * rms).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (a * w2).to(tl.bfloat16)

    # ReLU
    zero = tl.zeros_like(out)
    out = tl.maximum(out, zero)

    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_rms_relu_kernel[(Mrows,)](
            x, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
