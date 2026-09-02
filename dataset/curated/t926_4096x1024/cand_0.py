import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 926
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_bias_relu_scale_softmax(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_row,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # x + b0 : bf16 op done in fp32 opmath, rounded back to bf16
    v = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    # relu (exact in any precision)
    v = tl.maximum(v, 0.0)
    # * scalar : fp32 opmath, rounded to bf16
    v = (v.to(tl.float32) * SCALE).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch's internal upcast for bf16)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, n_cols = x.shape[0], x.shape[-1]
        x2 = x.view(-1, n_cols)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_bias_relu_scale_softmax[(rows,)](
            x2, self.b0, out,
            n_cols, x2.stride(0),
            SCALE=1.0557,
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return out.view_as(x)
