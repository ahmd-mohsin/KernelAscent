import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 844
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _ln_bias_relu_kernel(
    X, G, B, B2, B3, Y,
    stride_x, stride_y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptr = X + row * stride_x + cols
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    # mean / variance in fp32
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # layer_norm output rounded to bf16 (matches PyTorch bf16 layer_norm)
    y = (xc * rstd * g + b).to(tl.bfloat16)

    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)

    # bf16 add semantics: fp32 compute, round to bf16 each step
    y = (y.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)

    yf = y.to(tl.float32)
    yf = tl.where(yf > 0.0, yf, 0.0)
    y = (yf * scale).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_bias_relu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.b2, self.b3, out,
            h.stride(0), out.stride(0),
            N, 1e-5, 1.4218,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
