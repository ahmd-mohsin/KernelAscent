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
    X, OUT, B1, B3, G, B,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, round to fp16 like the reference output)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float16)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float16)

    # elementwise chain, rounding at each step to match fp16 reference
    y = (sm.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    y = (y.to(tl.float32) * 1.2334).to(tl.float16)
    y = (y.to(tl.float32) + b3.to(tl.float32)).to(tl.float16)

    # layer norm in fp32 (matches PyTorch internal fp32 for fp16 inputs)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd) * g + b

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = y + self.b1
            y = y * 1.2334
            y = y + self.b3
            return F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_affine_ln[(rows,)](
            x2, out, self.b1, self.b3, self.ln4_g, self.ln4_b,
            x2.stride(0), out.stride(0),
            N, BLOCK=BLOCK, EPS=1e-5,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
