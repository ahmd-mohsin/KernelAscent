import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 566
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_rms_gelu_kernel(
    X, W_RMS, B2, B4, OUT,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.float16)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)

    # scale + bias (fp32 compute, round to fp16 to match torch half elementwise)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    y = (y.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)

    # exact GELU (erf-based) in fp32, cast to fp16
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    y = g.to(tl.float16)

    y = (y.to(tl.float32) + b4.to(tl.float32)).to(tl.float16)

    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    y = g.to(tl.float16)

    tl.store(OUT + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        _fused_rms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, self.b4, out,
            N, 1e-6,
            BLOCK=512,
            num_warps=4,
        )
        return out
