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
    X, B, W2, W4, OUT,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)                   # fp16

    # bias add in fp16 (matches x + b1 in half precision)
    xh = (x + b).to(tl.float16)
    xf = xh.to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    t = (xf * inv).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    y = (t * w2).to(tl.float16)

    # ReLU (fp16)
    y = tl.maximum(y, 0.0).to(tl.float16)

    # RMSNorm 2
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + EPS)
    t2 = (yf * inv2).to(tl.float16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    out = (t2 * w4).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS fp16 GEMM
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
