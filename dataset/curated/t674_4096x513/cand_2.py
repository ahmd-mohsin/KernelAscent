import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 674
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, W_RMS, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X + row * stride_x + cols)          # fp16
    xf = x.to(tl.float32)

    # RMSNorm (compute in fp32, cast to fp16, multiply by weight in fp16)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS_RMS)
    t16 = (xf * inv).to(tl.float16)
    w = tl.load(W_RMS + cols)                        # fp16
    t16 = t16 * w                                    # fp16 multiply (matches ref)

    # LayerNorm (fp32 internals, like PyTorch on half input)
    tf = t16.to(tl.float32)
    mean = tl.sum(tf, axis=0) / N
    d = tf - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    y = d * rstd * g + b

    # ReLU + store as fp16
    y16 = y.to(tl.float16)
    zero = tl.zeros_like(y16)
    y16 = tl.maximum(y16, zero)
    tl.store(Y + row * stride_y + cols, y16)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _fused_norm_kernel[(Mrows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            x.stride(0), y.stride(0),
            N=N, EPS_RMS=1e-6, EPS_LN=1e-5,
            BLOCK=512,
            num_warps=4,
        )
        return y
