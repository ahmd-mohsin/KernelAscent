import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 254
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _ln_scale_kernel(
    X, G, B, Y,
    N, stride_x, stride_y,
    eps,
    s1, s2,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    # round to bf16 (layer_norm output), then two scalar muls each rounded to bf16
    y = y.to(tl.bfloat16)
    y = (y.to(tl.float32) * s1).to(tl.bfloat16)
    y = (y.to(tl.float32) * s2).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _ln_scale_kernel[(M_,)](
            h, self.ln1_g, self.ln1_b, out,
            N_, h.stride(0), out.stride(0),
            1e-5,
            1.2259, 1.4311,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
