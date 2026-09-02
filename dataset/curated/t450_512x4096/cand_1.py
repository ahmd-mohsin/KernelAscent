import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 450
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_softmax_affine_ln(
    X, B1, B3, G, B, Y,
    D: tl.constexpr,
    eps,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch half softmax)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    sm = e / s
    # round to fp16 (output of torch.softmax on half tensor)
    sm = sm.to(tl.float16).to(tl.float32)

    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b1  (fp32 add of fp16 values == fp16 add with RN after rounding)
    t = (sm + b1).to(tl.float16).to(tl.float32)
    # x = x * 1.2334 (opmath fp32, round to fp16)
    t = (t * scale).to(tl.float16).to(tl.float32)
    # x = x + b3
    t = (t + b3).to(tl.float16).to(tl.float32)

    # layer norm with fp32 statistics
    t_masked = tl.where(mask, t, 0.0)
    mu = tl.sum(t_masked, 0) / D
    diff = tl.where(mask, t - mu, 0.0)
    var = tl.sum(diff * diff, 0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    y = (t - mu) * rstd * g + b
    tl.store(Y + row * D + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_affine_ln[(rows,)](
            x, self.b1, self.b3, self.ln4_g, self.ln4_b, y,
            d, 1e-5, 1.2334,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
