import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 614
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_bias_scale_softmax(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_x, stride_out,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # match reference numerics: fp16 add, fp16 mul, then softmax (torch upcasts internally)
    v = (x + b).to(tl.float16)
    v = (v * scale).to(tl.float16)
    v = v.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))

    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_out + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, n_cols)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_bias_scale_softmax[(x2.shape[0],)](
            x2, self.b0, out,
            n_cols,
            x2.stride(0), out.stride(0),
            1.4684,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view_as(x)
