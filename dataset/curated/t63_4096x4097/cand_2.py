import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 63
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_kernel(X, B0, W2, W3, B4, Y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row.to(tl.int64) * n_cols

    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0 (bf16 result)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # exact GELU with fp32 opmath, rounded to bf16
    x = (0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    x = (x * w3).to(tl.bfloat16).to(tl.float32)

    # final bias
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x + b4).to(tl.bfloat16)
    tl.store(Y + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(n_rows,)](
            x2d, self.b0, self.rms2_w, self.rms3_w, self.b4, y,
            n_cols, BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
