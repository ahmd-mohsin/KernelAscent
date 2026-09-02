import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 801
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Y,
    N,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch's bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    # Round to bf16 exactly as the reference does between the two norms
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    # RMSNorm on the bf16-rounded values (fp32 math)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps_rms)

    # (_xf * rsqrt).to(bf16) * rms2_w  -> two roundings, replicate both
    z_bf = (yf * r).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z_bf.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_rms_kernel[(Mrows,)](
            x, self.ln1_g, self.ln1_b, self.rms2_w, y,
            N, 1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
