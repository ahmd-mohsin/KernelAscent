import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 148
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_softmax_ln_ln(
    X, G3, B3, G4, B4, OUT,
    stride_x, stride_o,
    N, scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)
    x = x * scale

    # softmax (fp32 accumulation, output rounded to bf16 like PyTorch)
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    e = tl.where(mask, e, 0.0)
    s = e / tl.sum(e, 0)
    s = s.to(tl.bfloat16).to(tl.float32)

    inv_n = 1.0 / N

    # layer norm 1
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.sum(s, 0) * inv_n
    d1 = tl.where(mask, s - mean1, 0.0)
    var1 = tl.sum(d1 * d1, 0) * inv_n
    y = d1 * (1.0 / tl.sqrt(var1 + eps)) * g3 + b3
    y = y.to(tl.bfloat16).to(tl.float32)

    # layer norm 2
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    mean2 = tl.sum(tl.where(mask, y, 0.0), 0) * inv_n
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) * inv_n
    z = d2 * (1.0 / tl.sqrt(var2 + eps)) * g4 + b4

    tl.store(OUT + row * stride_o + offs, z.to(tl.bfloat16), mask=mask)


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
        # matmul on tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_ln_ln[(m,)](
            h, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            h.stride(0), out.stride(0),
            n, 1.4775, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
