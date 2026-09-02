import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 638
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, Out,
    stride_x, stride_o,
    N, eps,
    s1, s2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    # x * 1.1576 in bf16 semantics (opmath fp32, round to bf16)
    xf = x.to(tl.float32) * s1
    xb = xf.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, output rounded to bf16
    xb = tl.where(mask, xb, float('-inf'))
    m = tl.max(xb, axis=0)
    e = tl.exp(xb - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # * 1.3589 (opmath fp32, round to bf16)
    y = (sm.to(tl.float32) * s2).to(tl.bfloat16)

    # layer norm in fp32
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / N
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = d * rstd * g + b

    # relu (applied after rounding to bf16)
    out = out.to(tl.bfloat16)
    out = tl.maximum(out, 0.0)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.ln3_g, self.ln3_b, out,
            x2.stride(0), out.stride(0),
            N, 1e-5,
            1.1576, 1.3589,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
