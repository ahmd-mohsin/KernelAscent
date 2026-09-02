import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 862
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _ln_scale_gelu_kernel(
    X, OUT, G, B,
    N, stride_x, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    # round to fp16 to match reference intermediate dtype
    y = y.to(tl.float16).to(tl.float32)
    y = y * scale
    y = y.to(tl.float16).to(tl.float32)

    # exact GELU: y * 0.5 * (1 + erf(y / sqrt(2)))
    out = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        out = torch.empty_like(x)
        _ln_scale_gelu_kernel[(Mrows,)](
            x, out, self.ln1_g, self.ln1_b,
            N, x.stride(0), out.stride(0),
            1.4752,
            BLOCK=512,
            num_warps=4,
        )
        return out
