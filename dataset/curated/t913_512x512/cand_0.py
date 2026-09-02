import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 913
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr, n_cols, stride_x, stride_o,
                  eps, s1, s2, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.3419 (bf16 rounding)
    x = (x * s1).to(tl.bfloat16).to(tl.float32)
    # x = x * 1.3693 (bf16 rounding)
    x = (x * s2).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), computed in fp32, rounded to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / n_cols
    inv = tl.math.rsqrt(ms + eps)
    y = (g * inv).to(tl.bfloat16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    out = y * w

    tl.store(out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        w = self.rms3_w
        if not w.is_cuda:
            w = w.cuda()
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(n_rows,)](
            x2, w, out, n_cols,
            x2.stride(0), out.stride(0),
            1e-6, 1.3419, 1.3693,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)
