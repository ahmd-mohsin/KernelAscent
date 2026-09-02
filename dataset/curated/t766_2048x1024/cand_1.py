import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 766
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_rms_ln_gelu(
    X, RW, G, B, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)  # bf16
    xf = x.to(tl.float32)

    # RMSNorm (fp32 math, cast to bf16 before weight mul, like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    y_bf = (xf * rrms).to(tl.bfloat16)

    rw = tl.load(RW + cols, mask=mask, other=0.0)  # bf16
    z_bf = y_bf * rw  # bf16 multiply, matches reference rounding
    z = z_bf.to(tl.float32)

    # LayerNorm (fp32 opmath like PyTorch)
    mean = tl.sum(z, axis=0) / N
    zc = tl.where(mask, z - mean, 0.0)
    var = tl.sum(zc * zc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h = zc * rstd * g + b

    # exact GELU (erf), fp32 opmath
    out = 0.5 * h * (1.0 + tl.math.erf(h * 0.7071067811865476))

    tl.store(Y + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _fused_rms_ln_gelu[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return y
