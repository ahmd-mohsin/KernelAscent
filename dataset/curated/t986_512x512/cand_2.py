import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 986
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_rms_relu_bias_softmax(
    X, W, B, OUT,
    N, stride_x, stride_o,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 math, round to fp16, then multiply by weight)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # ReLU
    y = tl.maximum(y, 0.0)

    # + bias (fp16 rounding semantics via fp32 opmath)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # * scale
    y = (y.to(tl.float32) * scale).to(tl.float16)

    # Softmax in fp32 (matches torch half softmax accumulation)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float("-inf"))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_relu_bias_softmax[(m,)](
            x, self.rms1_w, self.b3, out,
            n, x.stride(0), out.stride(0),
            1.0065, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
