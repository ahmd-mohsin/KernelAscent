import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 972
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_softmax_ln_kernel(
    X, W, B, Y,
    N,
    stride_xm, stride_ym,
    EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round back to bf16 like the eager op
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scalar multiply (fp32 opmath, rounded to bf16)
    s = g * 1.2255
    s = s.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    s = tl.where(mask, s, float('-inf'))
    mx = tl.max(s, axis=0)
    e = tl.exp(s - mx)
    denom = tl.sum(e, axis=0)
    p = e / denom
    # softmax output is materialized as bf16 in the reference
    p = p.to(tl.bfloat16).to(tl.float32)
    p = tl.where(mask, p, 0.0)

    # layer norm in fp32
    mean = tl.sum(p, axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * w + b
    tl.store(Y + row * stride_ym + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = y * 1.2255
            y = torch.softmax(y, dim=-1)
            return F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_gelu_softmax_ln_kernel[(rows,)](
            x2, self.ln3_g, self.ln3_b, y,
            N,
            x2.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
