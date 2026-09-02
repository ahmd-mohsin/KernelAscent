import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 581
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X, W, Y,
    N,
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches torch's fp32 accumulation for bf16 softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # round to bf16 then upcast (matches x = softmax(...).to(bf16); _xf = x.float())
    s_bf = s.to(tl.bfloat16)
    xf = s_bf.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = xn * w

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_rms_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
