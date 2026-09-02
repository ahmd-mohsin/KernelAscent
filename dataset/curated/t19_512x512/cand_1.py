import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 19
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_double_rms_kernel(
    X_ptr, W1_ptr, W3_ptr, Y_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load input row (bf16 -> f32)
    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.2975  (opmath in f32, result rounded to bf16)
    x = (x * 1.2975).to(tl.bfloat16).to(tl.float32)

    # first RMS norm (in f32), then cast to bf16
    ms1 = tl.sum(x * x, axis=0) / D
    r1 = tl.math.rsqrt(ms1 + 1e-6)
    xn = (x * r1).to(tl.bfloat16).to(tl.float32)

    # multiply by rms1_w (bf16*bf16 with f32 opmath, rounded to bf16)
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w1).to(tl.bfloat16).to(tl.float32)

    # relu
    y = tl.maximum(y, 0.0)

    # second RMS norm (in f32), then cast to bf16
    ms2 = tl.sum(y * y, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    yn = (y * r2).to(tl.bfloat16).to(tl.float32)

    # multiply by rms3_w, store bf16
    w3 = tl.load(W3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (yn * w3).to(tl.bfloat16)
    tl.store(Y_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference path (CPU fallback)
            x = x * 1.2975
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        x = x.contiguous()
        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        _fused_double_rms_kernel[(m,)](
            x2d, self.rms1_w, self.rms3_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
