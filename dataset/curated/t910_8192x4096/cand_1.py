import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 910
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_relu_bias_relu_ln(
    X, B, G, Beta, Y,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # relu -> +bias -> relu (bias add done in fp16 to match reference numerics)
    x = tl.maximum(x, 0.0)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x.to(tl.float16) + b.to(tl.float16)).to(tl.float32)
    x = tl.maximum(x, 0.0)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + beta

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        y = torch.empty_like(h)
        Mrows, N = h.shape
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_bias_relu_ln[(Mrows,)](
            h, self.b2, self.ln4_g, self.ln4_b, y,
            h.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
