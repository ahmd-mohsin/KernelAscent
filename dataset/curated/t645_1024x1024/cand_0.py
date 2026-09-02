import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 645
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_relu_ln_relu_ln(
    X, Y, G2, B2, G4, B4,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)

    # layer norm 1
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x - mean) * rstd * g2 + b2

    # relu
    x = tl.maximum(x, 0.0)

    # layer norm 2
    mean2 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(diff2 * diff2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean2) * rstd2 * g4 + b4

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_ln_relu_ln[(m,)](
            h, out, self.ln2_g, self.ln2_b, self.ln4_g, self.ln4_b,
            h.stride(0), out.stride(0),
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
