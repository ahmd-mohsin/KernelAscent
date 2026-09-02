import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 468
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_scale_double_ln(
    X, Y, G1, B1, G2, B2,
    D: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.3642 (computed in fp32, rounded to fp16 like PyTorch)
    x = (x * 1.3642).to(tl.float16).to(tl.float32)

    # ----- LayerNorm 1 -----
    mean1 = tl.sum(x, axis=0) / D
    diff1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(diff1 * diff1, axis=0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = diff1 * rstd1 * g1 + b1
    # round to fp16 (LN1 output dtype in the reference), then back to fp32
    y1 = y1.to(tl.float16).to(tl.float32)
    y1 = tl.where(mask, y1, 0.0)

    # ----- LayerNorm 2 -----
    mean2 = tl.sum(y1, axis=0) / D
    diff2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(diff2 * diff2, axis=0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = diff2 * rstd2 * g2 + b2

    tl.store(Y + row * D + cols, y2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.3642
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x

        x = x.contiguous()
        rows = x.numel() // x.shape[-1]
        d = x.shape[-1]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_scale_double_ln[(rows,)](
            x, y,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            d, 1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
