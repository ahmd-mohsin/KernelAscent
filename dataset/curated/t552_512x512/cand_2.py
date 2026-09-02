import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 552
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _rms_softmax_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    normed = xf * inv

    # cast to bf16 (match reference .to(x.dtype)), multiply by bf16 weight -> bf16
    w = tl.load(W + cols, mask=mask, other=0.0)
    y_bf16 = (normed.to(tl.bfloat16) * w).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch bf16 softmax with fp32 accumulation)
    yf = y_bf16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _rms_softmax_kernel[(m,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
