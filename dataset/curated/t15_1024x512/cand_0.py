import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 15
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_bias_double_ln(
    X, Y, B0, G1, B1, G2, B2,
    stride_x, stride_y,
    N, EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0 (rounded to fp16 as in reference)
    x = (x + b0).to(tl.float16).to(tl.float32)

    # LayerNorm 1
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    bb1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + bb1
    y1 = y1.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(y1, axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    bb2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = d2 * rstd2 * g2 + bb2

    tl.store(Y + row * stride_y + cols, y2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_bias_double_ln[(rows,)](
            x2d, y, self.b0, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            x2d.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
