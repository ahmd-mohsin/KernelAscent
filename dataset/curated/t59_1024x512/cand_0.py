import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 59
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Out,
    N, stride_row,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch on bf16 inputs)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = xc * rstd * g + b
    # cast to bf16 (LN output dtype), then back to fp32 for RMS (matches _xf = x.float())
    ln_bf16 = ln.to(tl.bfloat16)
    xf = ln_bf16.to(tl.float32)

    # RMSNorm
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    t = (xf * rrms).to(tl.bfloat16)  # .to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 elementwise (fp32 multiply, round to bf16)
    out = (t.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    tl.store(Out + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 matmul
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_rms_kernel[(Mrows,)](
            y, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N, y.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
