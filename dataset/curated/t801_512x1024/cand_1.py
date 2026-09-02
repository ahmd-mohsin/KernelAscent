import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 801
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _ln_rms_kernel(
    X, G, B, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, like PyTorch's bf16 layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = xc * rstd * g + b

    # round to bf16 (reference produces bf16 output of layer_norm)
    ln_bf16 = ln.to(tl.bfloat16)
    xf = ln_bf16.to(tl.float32)

    # ---- RMSNorm in fp32 ----
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)
    y_bf16 = (xf * rrms).to(tl.bfloat16)

    # bf16 * bf16 multiply (compute in f32, round to bf16) matches GPU bf16 mul
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y_bf16.to(tl.float32) * w).to(tl.bfloat16)

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
        h = x @ self.W0  # cuBLAS bf16 matmul, identical to reference
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_rms_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N=N,
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
