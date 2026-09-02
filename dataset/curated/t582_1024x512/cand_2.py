import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 582
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _ln_relu_gelu_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    # round to bf16 (layer_norm output), then relu
    y = y.to(tl.bfloat16)
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)
    # gelu (erf, exact) computed in fp32 from bf16 input
    yf = y.to(tl.float32)
    out = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        shape = x.shape
        N = shape[-1]
        Mrows = x.numel() // N
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _ln_relu_gelu_kernel[(Mrows,)](
            x, self.ln1_g, self.ln1_b, y,
            N, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y
