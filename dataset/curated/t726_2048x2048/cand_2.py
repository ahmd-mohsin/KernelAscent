import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 726
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_softmax_kernel(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # x * 1.0428 in bf16 semantics (compute fp32, round to bf16)
    xs = (x.to(tl.float32) * 1.0428).to(tl.bfloat16)
    # add bias in bf16 semantics
    xb = (xs.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch internal upcast)
    v = xb.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_kernel[(n_rows,)](
            x2, self.b1, out,
            n_cols,
            x2.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
