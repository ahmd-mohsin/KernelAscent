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
    N, stride_row,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x16 = tl.load(X + row * stride_row + offs, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm (mean of squares over the row, computed in fp32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    # normalize -> round to fp16 (matches .to(x.dtype))
    y = (xf * r).to(tl.float16)

    # * rms1_w : half op with float opmath, rounded to half
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # relu
    y = tl.maximum(y, 0.0)

    # + b3 : float opmath, round to half
    b = tl.load(B + offs, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # * 1.0065 : float opmath, round to half
    y = (y.to(tl.float32) * scale).to(tl.float16)

    # softmax in fp32 (matches torch's half softmax with float accumulation)
    v = tl.where(mask, y.to(tl.float32), float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # tensor-core GEMM
        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_relu_bias_softmax[(rows,)](
            x, self.rms1_w, self.b3, out,
            N, x.stride(0),
            1e-6, 1.0065,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
