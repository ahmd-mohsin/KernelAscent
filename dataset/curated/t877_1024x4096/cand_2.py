import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 877
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_softmax_ln_ln_relu(
    X, Y, G1, B1, G2, B2,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 math, round to bf16 like reference)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm 1
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    x = d1 * rstd1 * g1 + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm 2
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    mean2 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    x = d2 * rstd2 * g2 + b2

    # relu
    x = tl.maximum(x, 0.0)

    tl.store(Y + row * stride_y + cols, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return torch.relu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_ln_ln_relu[(rows,)](
            x2d, y,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            x2d.stride(0), y.stride(0),
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
