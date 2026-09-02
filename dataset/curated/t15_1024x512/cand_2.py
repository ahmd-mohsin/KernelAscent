import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 15
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_kernel(X, B0, G1, Bt1, G2, Bt2, Y,
                  N, D: tl.constexpr, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # fp16 add (match reference: x + b0 in fp16)
    x = (x + b0).to(tl.float16)

    xf = x.to(tl.float32)

    # LayerNorm 1 (fp32 accumulation)
    mean1 = tl.sum(xf, axis=0) / D
    d1 = tl.where(mask, xf - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    bt1 = tl.load(Bt1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + bt1

    # cast intermediate to fp16 (match reference dtype boundary)
    y1 = y1.to(tl.float16).to(tl.float32)

    # LayerNorm 2 (fp32 accumulation)
    mean2 = tl.sum(tl.where(mask, y1, 0.0), axis=0) / D
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    bt2 = tl.load(Bt2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = d2 * rstd2 * g2 + bt2

    tl.store(Y + row * D + cols, y2.to(tl.float16), mask=mask)


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

        x = x.contiguous()
        n_rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(n_rows,)](
            x, self.b0, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, y,
            n_rows, d, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
