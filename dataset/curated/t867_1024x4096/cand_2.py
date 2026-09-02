import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 867
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_bias_ln_bias(
    X, B0, G, B, B2, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b0

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # match reference: layer_norm output cast to bf16, then bf16 add of b2
    y = y.to(tl.bfloat16).to(tl.float32) + b2

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x = x.contiguous().view(-1, N)
        Mrows = x.shape[0]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4
        _fused_bias_ln_bias[(Mrows,)](
            x, self.b0, self.ln1_g, self.ln1_b, self.b2, y,
            N, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
