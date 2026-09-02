import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 752
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_relu_ln_bias_scale(
    X, G, B, B4, Y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # relu(relu(x)) == relu(x)
    x = tl.maximum(x, 0.0)

    # layer norm statistics in fp32 (matches PyTorch bf16 layer_norm internals)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # round to bf16 like layer_norm output, then do (+ b4) and (* scale)
    # each in fp32 with bf16 rounding between ops (matches eager op chain)
    y = y.to(tl.bfloat16).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b4).to(tl.bfloat16).to(tl.float32)
    y = (y * scale).to(tl.bfloat16)

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_relu_ln_bias_scale[(Mrows,)](
            h, self.ln3_g, self.ln3_b, self.b4, y,
            N, 1e-5, 1.4319,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
