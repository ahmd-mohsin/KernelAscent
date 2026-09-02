import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 891
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _softmax_ln_kernel(
    X, Y, G, B,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's half softmax)
    xmax = tl.max(x, axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to fp16 (softmax output dtype), then back to fp32 for layernorm
    p16 = p.to(tl.float16)
    p = p16.to(tl.float32)

    # layernorm in fp32
    mean = tl.sum(p, axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _softmax_ln_kernel[(m,)](
            h, out, self.ln2_g, self.ln2_b,
            h.stride(0), out.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
