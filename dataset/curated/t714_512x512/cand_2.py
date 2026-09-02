import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 714
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_epilogue(X, B, G, Bt, Rw, Out,
                    N, stride_x, stride_o,
                    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)                   # fp16

    # bias add + relu in fp16 (matches reference)
    x = x + b
    x = tl.maximum(x, 0.0)

    # layernorm in fp32
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(Bt + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + bt
    y16 = y.to(tl.float16)  # cast to fp16 as in reference output of layer_norm

    # rmsnorm: upcast to fp32
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    z16 = (yf * rrms).to(tl.float16)

    rw = tl.load(Rw + cols, mask=mask, other=0.0)  # fp16
    out = z16 * rw  # fp16 multiply, matches reference

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue[(Mrows,)](
            h, self.b1, self.ln3_g, self.ln3_b, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
