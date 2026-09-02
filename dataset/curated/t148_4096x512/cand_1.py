import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 148
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_scale_softmax_ln_ln(
    X, OUT, G3, B3, G4, B4,
    stride_x, stride_o,
    N: tl.constexpr, SCALE: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # scale (rounded to bf16 to match PyTorch elementwise op)
    x = (x.to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    x_m = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x_m, axis=0)
    num = tl.exp(x_m - row_max)
    num = tl.where(mask, num, 0.0)
    den = tl.sum(num, axis=0)
    sm = (num / den).to(tl.bfloat16).to(tl.float32)

    # layernorm 1
    mean1 = tl.sum(sm, axis=0) / N
    d1 = tl.where(mask, sm - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = (d1 * rstd1 * g3 + b3).to(tl.bfloat16).to(tl.float32)

    # layernorm 2
    mean2 = tl.sum(tl.where(mask, y1, 0.0), axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = (d2 * rstd2 * g4 + b4).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.matmul(x, self.W0)  # tensor-core bf16 matmul
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        grid = (m,)
        _fused_scale_softmax_ln_ln[grid](
            y, out,
            self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b,
            y.stride(0), out.stride(0),
            N=n, SCALE=1.4775, EPS=1e-5,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
