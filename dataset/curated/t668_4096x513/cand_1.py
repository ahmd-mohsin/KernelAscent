import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 668
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _softmax_rms_kernel(
    X, W, OUT,
    stride_x, stride_o,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's internal fp32 accumulation for fp16 input)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (softmax output dtype), then back to fp32 like x.float()
    p16 = p.to(tl.float16)
    xf = p16.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)

    y = (xf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = y * w  # fp16 multiply, same as reference

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _softmax_rms_kernel[(m,)](
            x2, self.rms1_w, out,
            x2.stride(0), out.stride(0),
            d, BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return out.view(orig_shape)
