import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 118
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, D: tl.constexpr, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bf16 add (match reference: bf16 + bf16 -> bf16)
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    xb = tl.where(mask, xb, float('-inf'))

    # first softmax in fp32, round to bf16 (matches PyTorch bf16 softmax output)
    m1 = tl.max(xb, axis=0)
    e1 = tl.exp(xb - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = (e1 / s1).to(tl.bfloat16)

    # relu (identity on nonnegative softmax outputs)
    y1 = tl.maximum(y1, 0.0)

    # second softmax on bf16 values, fp32 accumulation
    y1f = tl.where(mask, y1.to(tl.float32), float('-inf'))
    m2 = tl.max(y1f, axis=0)
    e2 = tl.exp(y1f - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        orig_shape = x.shape
        x2 = x.view(-1, d)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(x2.shape[0],)](
            x2, self.b0, y, d, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return y.view(orig_shape)
