import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 441
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_double_softmax_kernel(
    x_ptr, b1_ptr, b2_ptr, out_ptr,
    n_cols,
    stride_x, stride_out,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, like torch on fp16 input)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = e1 / s1
    # round to fp16 (softmax output dtype in reference)
    y = y.to(tl.float16)

    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    # elementwise adds: computed in fp32 (opmath), rounded to fp16 each step
    y = (y.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    y = (y.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)

    z = y.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))

    # softmax 2
    m2 = tl.max(z, axis=0)
    e2 = tl.exp(z - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    w = (e2 / s2).to(tl.float16)

    # scalar multiply: fp32 compute, fp16 output
    out = (w.to(tl.float32) * SCALE).to(tl.float16)
    tl.store(out_ptr + row * stride_out + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = y + self.b1
            y = y + self.b2
            y = torch.softmax(y, dim=-1)
            return y * 1.2172

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_double_softmax_kernel[(n_rows,)](
            x2, self.b1, self.b2, out,
            n_cols,
            x2.stride(0), out.stride(0),
            SCALE=1.2172,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
