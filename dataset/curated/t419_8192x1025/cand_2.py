import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 419
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_bias_rms_softmax(
    x_ptr, b_ptr, w_ptr, out_ptr,
    D, stride_x, stride_o, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0  (bf16 add == fp32 add then round to bf16)
    xb = (x + b).to(tl.bfloat16)

    # RMSNorm in fp32, cast back to bf16, then multiply by bf16 weight
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + eps)
    normed = (xf * inv).to(tl.bfloat16)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = normed * w  # bf16 * bf16 -> bf16 (single rounding, matches torch)

    # Softmax in fp32 (matches torch's acc-type softmax for bf16)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    e = tl.math.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + offs, p, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape
        p = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D_)
        _fused_bias_rms_softmax[(M_,)](
            x, self.b0, self.rms1_w, p,
            D_, x.stride(0), p.stride(0), 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return p @ self.W3
