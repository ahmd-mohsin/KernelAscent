import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 827
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b0_ptr, b2_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(b2_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # x = x + b0  (fp16 rounding to match reference)
    t = (x + b0).to(tl.float16).to(tl.float32)
    # gelu (exact erf form, computed in fp32, rounded to fp16)
    t = (t * 0.5 * (1.0 + tl.math.erf(t * INV_SQRT2))).to(tl.float16).to(tl.float32)
    # x = x + b2
    t = (t + b2).to(tl.float16).to(tl.float32)
    # gelu
    t = (t * 0.5 * (1.0 + tl.math.erf(t * INV_SQRT2))).to(tl.float16).to(tl.float32)
    # scale
    t = (t * 1.2166).to(tl.float16).to(tl.float32)

    # softmax over row (fp32 accumulation, like PyTorch half softmax)
    t = tl.where(mask, t, float('-inf'))
    row_max = tl.max(t, axis=0)
    e = tl.exp(t - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = (e / denom).to(tl.float16)

    tl.store(out_ptr + row * n_cols + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, cols = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        x2 = x.view(rows, cols)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.b0, self.b2, out, cols,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view_as(x)
