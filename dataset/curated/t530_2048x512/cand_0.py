import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 530
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_softmax_rms_rms_ln(
    X_ptr, B1_ptr, W3_ptr, W4_ptr, G5_ptr, B5_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # load row (fp16) and bias (fp16); bias add in fp16 to match reference
    x_h = tl.load(X_ptr + row * stride_x + offs)
    b_h = tl.load(B1_ptr + offs)
    x_h = x_h + b_h  # fp16 add, matches `x = x + self.b1`

    # softmax computed in fp32 (matches PyTorch half softmax opmath), stored back to fp16
    xf = x_h.to(tl.float32)
    mmax = tl.max(xf, 0)
    e = tl.exp(xf - mmax)
    s = tl.sum(e, 0)
    x_h = (e / s).to(tl.float16)

    n_f = N.to(tl.float32)

    # RMSNorm 3
    xf = x_h.to(tl.float32)
    r = tl.math.rsqrt(tl.sum(xf * xf, 0) / n_f + EPS_RMS)
    w3 = tl.load(W3_ptr + offs)
    x_h = (xf * r).to(tl.float16) * w3  # fp16 multiply, matches reference

    # RMSNorm 4
    xf = x_h.to(tl.float32)
    r = tl.math.rsqrt(tl.sum(xf * xf, 0) / n_f + EPS_RMS)
    w4 = tl.load(W4_ptr + offs)
    x_h = (xf * r).to(tl.float16) * w4

    # LayerNorm (fp32 internals, matches native half layer_norm)
    xf = x_h.to(tl.float32)
    mu = tl.sum(xf, 0) / n_f
    d = xf - mu
    var = tl.sum(d * d, 0) / n_f
    rstd = tl.math.rsqrt(var + EPS_LN)
    g = tl.load(G5_ptr + offs).to(tl.float32)
    b = tl.load(B5_ptr + offs).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_rms_rms_ln[(m,)](
            h, self.b1, self.rms3_w, self.rms4_w, self.ln5_g, self.ln5_b, y,
            h.stride(0), y.stride(0),
            n,
            EPS_RMS=1e-6,
            EPS_LN=1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
