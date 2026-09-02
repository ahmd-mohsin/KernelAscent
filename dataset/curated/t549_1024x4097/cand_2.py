import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 549
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _ln_bias_gelu_kernel(
    X, G, B, B2, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # mean / variance in fp32
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # round to bf16 as layer_norm output would be, then fp32 add
    y = y.to(tl.bfloat16).to(tl.float32)
    y = y + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5*x*(1+erf(x/sqrt(2)))
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS bf16 tensor-core matmul
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _ln_bias_gelu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.b2, y,
            N, 1e-5,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y
