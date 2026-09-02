import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 521
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b1_ptr, w_ptr, out_ptr,
    N,  # row length
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.405 (bf16 rounding to match reference)
    x = (x * 1.405).to(tl.bfloat16).to(tl.float32)
    # x = x + b1 (bf16 rounding)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)

    # softmax in fp32 accumulation, output rounded to bf16
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(sm * sm, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    normed = (sm * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (normed * w).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x * 1.405
            x = x + self.b1
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, N = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2, self.b1, self.rms4_w, out,
            N, x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4 if BLOCK <= 1024 else 8,
        )
        return out.view(orig_shape)
