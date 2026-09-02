import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 417
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_bias_rms_kernel(
    x_ptr, b0_ptr, b1_ptr, b2_ptr, w_ptr, out_ptr,
    D: tl.constexpr, stride_row, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    # Sequential bf16 adds (round to bf16 after each add, matching PyTorch semantics)
    x = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    r = 1.0 / tl.sqrt(ms + eps)

    y = (xf * r).to(tl.bfloat16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_bias_rms_kernel[(rows,)](
            x, self.b0, self.b1, self.b2, self.rms3_w, out,
            d, x.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
