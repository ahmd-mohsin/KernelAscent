import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 399
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_bias_ln_rms_kernel(
    X, B, G, LB, W, Out,
    N, stride_x, stride_o,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b

    # LayerNorm (fp32 math, like PyTorch's fp16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    lb = tl.load(LB + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + lb

    # cast to fp16 (layer_norm output dtype), then back to fp32 for RMS
    y16 = y.to(tl.float16)
    yf = y16.to(tl.float32)

    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + eps_rms)

    z16 = (yf * rrms).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = z16 * w  # fp16 multiply, matching reference

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_bias_ln_rms_kernel[(Mrows,)](
            h, self.b1, self.ln2_g, self.ln2_b, self.rms3_w, out,
            N, h.stride(0), out.stride(0),
            1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
