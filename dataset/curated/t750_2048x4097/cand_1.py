import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 750
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _fused_bias_relu_rms_kernel(
    x_ptr, b0_ptr, w_ptr, b4_ptr, out_ptr,
    D, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    o_row = out_ptr + row * stride_o

    # Pass 1: accumulate sum of squares of relu(x + b0) in fp32
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0)      # fp16
        b = tl.load(b0_ptr + cols, mask=mask, other=0.0)     # fp16
        v = tl.maximum(x + b, 0.0)                            # fp16 add (matches ref)
        vf = v.to(tl.float32)
        acc += vf * vf
    ssum = tl.sum(acc, axis=0)
    inv = 1.0 / tl.sqrt(ssum / D + eps)

    # Pass 2: normalize, scale, add bias, store
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0)
        b = tl.load(b0_ptr + cols, mask=mask, other=0.0)
        v = tl.maximum(x + b, 0.0)                            # fp16
        vf = v.to(tl.float32)
        n = (vf * inv).to(tl.float16)                         # cast then fp16 ops (matches ref)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)       # fp16
        b4 = tl.load(b4_ptr + cols, mask=mask, other=0.0)     # fp16
        y = n * w + b4
        tl.store(o_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = x + self.b4
            return x

        x = x.contiguous()
        Mrows, Dcols = x.shape
        out = torch.empty_like(x)
        BLOCK = 2048
        grid = (Mrows,)
        _fused_bias_relu_rms_kernel[grid](
            x, self.b0, self.rms3_w, self.b4, out,
            Dcols, x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
