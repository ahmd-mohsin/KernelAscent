import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 730
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _double_ln_bias_kernel(
    X, Y, G1, B1, G2, B2, B3,
    N, stride_x, stride_y,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First LayerNorm (fp32 math, matching PyTorch mixed-precision LN)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1
    # cast to bf16 and back (intermediate tensor dtype in reference)
    y1 = y1.to(tl.bfloat16).to(tl.float32)

    # Second LayerNorm
    mean2 = tl.sum(y1, axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = (d2 * rstd2 * g2 + b2).to(tl.bfloat16)

    # bias add in bf16 (matching reference dtype arithmetic)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    out = y2 + b3

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _double_ln_bias_kernel[(Mrows,)](
            x, y,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.b3,
            N, x.stride(0), y.stride(0),
            1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
