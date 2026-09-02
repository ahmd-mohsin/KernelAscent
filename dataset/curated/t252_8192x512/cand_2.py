import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 252
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu_bias_ln_kernel(
    X, B1, G, B2, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32, then round to bf16 (matching PyTorch's bf16 output of F.gelu)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (g + b1).to(tl.bfloat16).to(tl.float32)

    # layer norm in fp32
    mean = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gam = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bet = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * gam + bet

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_bias_ln_kernel[(m,)](
            x, self.b1, self.ln2_g, self.ln2_b, y,
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
