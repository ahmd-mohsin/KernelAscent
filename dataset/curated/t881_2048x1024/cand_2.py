import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 881
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, stride_x, stride_y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.1029  (computed in fp32, rounded back to fp16 like PyTorch)
    x = (x * 1.1029).to(tl.float16).to(tl.float32)

    # softmax 1 (fp32 accumulate, fp16 output)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32, output fp16, then fp16 multiply by weight
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    xn = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    x = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16).to(tl.float32)

    # softmax 2
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # softmax 3
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.1029
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        rows, d = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(rows,)](
            x2d, self.rms2_w, y,
            x2d.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
