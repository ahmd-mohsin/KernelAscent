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
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / D
    xn = x * tl.math.rsqrt(ms + EPS)

    # round to input dtype (bf16) as reference does .to(x.dtype)
    xn = xn.to(Y.dtype.element_ty).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

    # bf16 elementwise mul in PyTorch uses fp32 opmath then rounds to bf16
    y = (xn * w).to(Y.dtype.element_ty).to(tl.float32)

    # softmax in fp32 (matches PyTorch internal accumulation for bf16)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + offs, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            xr = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            return torch.softmax(xr, dim=-1)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _rms_softmax_kernel[(m,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            D=d,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
