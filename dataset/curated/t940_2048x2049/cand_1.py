import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 940
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _bias_softmax_kernel(
    x_ptr, b0_ptr, b1_ptr, out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)

    # replicate bf16 add rounding: fp32 add then round to bf16 at each step
    t = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    t = (t.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)

    v = tl.where(mask, t.to(tl.float32), float("-inf"))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _bias_softmax_kernel[(rows,)](
            x2, self.b0, self.b1, out,
            N, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
