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
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # fused double relu == single relu
    x = tl.maximum(x, 0.0)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # round to bf16 as PyTorch layer_norm output
    y = y.to(tl.bfloat16).to(tl.float32)

    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b4)
    y = y.to(tl.bfloat16).to(tl.float32)  # bf16 add rounding
    y = y * SCALE

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core bf16 matmul
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _fused_relu_ln_bias_scale[(M_,)](
            h, self.ln3_g, self.ln3_b, self.b4, out,
            h.stride(0), out.stride(0),
            N=N_, EPS=1e-5, SCALE=1.4319,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
