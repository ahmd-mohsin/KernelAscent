import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 807
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_bias_rms_relu_rms(
    X, B1, W2, W4, OUT,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B1 + cols, mask=mask, other=0.0)

    # bias add in fp16 (matches torch fp16 add)
    x = (x + b).to(tl.float16)

    # RMSNorm 1
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    xn = (xf * inv).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    x = xn * w2  # fp16 multiply

    # ReLU (fp16)
    zero = tl.zeros_like(x)
    x = tl.maximum(x, zero)

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    xn = (xf * inv).to(tl.float16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    y = xn * w4

    tl.store(OUT + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core GEMM
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_rms_relu_rms[(m,)](
            h, self.b1, self.rms2_w, self.rms4_w, out,
            n, h.stride(0), out.stride(0),
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
