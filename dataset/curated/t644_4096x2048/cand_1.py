import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 644
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, B1, G3, Bb3, G4, Bb4, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, fp16 output like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    # add bias (fp16 arithmetic to match PyTorch)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float16)
    y = (y + b1).to(tl.float32)

    # exact GELU (erf), computed in fp32 then rounded to fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    eps: tl.constexpr = 1e-5

    # layernorm 1
    mean1 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d1 = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(Bb3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g3 + b3
    y = y.to(tl.float16).to(tl.float32)

    # layernorm 2
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(Bb4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g4 + b4

    tl.store(Out + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = y + self.b1
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(rows,)](
            x2, self.b1, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            x2.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
