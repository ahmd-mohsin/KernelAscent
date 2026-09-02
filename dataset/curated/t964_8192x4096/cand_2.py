import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 964
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_ln_bias_kernel(
    X, Y, G, B, B2,
    stride_xm, stride_ym,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch's accscalar behavior)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    # softmax output is materialized as fp16 in the reference; replicate rounding
    p = p.to(tl.float16).to(tl.float32)

    # layer norm (fp32 accumulation)
    mean = tl.sum(p, axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    # match reference: layernorm output rounded to fp16 before adding b2
    y = y.to(tl.float16).to(tl.float32) + b2

    tl.store(Y + row * stride_ym + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return x + self.b2

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_ln_bias_kernel[(Mrows,)](
            x2d, y, self.ln1_g, self.ln1_b, self.b2,
            x2d.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return y.view(orig_shape)
