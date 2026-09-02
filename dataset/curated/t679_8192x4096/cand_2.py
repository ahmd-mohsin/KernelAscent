import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 679
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X, W0, W1, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First RMSNorm (fp32 math, then cast to fp16, multiply by w0 in fp16)
    ms0 = tl.sum(x * x, axis=0) / D
    rstd0 = 1.0 / tl.sqrt(ms0 + EPS)
    h = (x * rstd0).to(tl.float16)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    h = h * w0  # fp16 multiply, matches PyTorch

    # Second RMSNorm
    hf = h.to(tl.float32)
    ms1 = tl.sum(hf * hf, axis=0) / D
    rstd1 = 1.0 / tl.sqrt(ms1 + EPS)
    o = (hf * rstd1).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    o = o * w1

    tl.store(Y + row * stride_y + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _double_rmsnorm_kernel[(m,)](
            x2, self.rms0_w, self.rms1_w, y,
            x2.stride(0), y.stride(0),
            D=d, EPS=1e-6, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
