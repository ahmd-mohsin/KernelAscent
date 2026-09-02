import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 989
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b1_ptr, g_ptr, b_ptr, w_ptr, out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    base = x_ptr + row * stride_row
    x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.4347  (bf16 elementwise -> compute fp32, round bf16)
    x = (x * 1.4347).to(tl.bfloat16).to(tl.float32)

    # x = x + b1
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)

    # gelu (exact, erf)
    inv_sqrt2 = 0.7071067811865476
    x = (0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))).to(tl.bfloat16).to(tl.float32)

    # layer_norm (stats in fp32)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = ((x - mean) * rstd * g + bb).to(tl.bfloat16).to(tl.float32)

    # rms norm: fp32 compute on bf16 x, cast bf16, then * w (bf16 op)
    xm2 = tl.where(mask, x * x, 0.0)
    ms = tl.sum(xm2, axis=0) / N
    rrms = tl.math.rsqrt(ms + 1e-6)
    xn = (x * rrms).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2, self.b1, self.ln3_g, self.ln3_b, self.rms4_w, out,
            N, x2.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
