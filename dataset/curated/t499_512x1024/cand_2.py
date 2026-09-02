import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 499
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, w2_ptr, w3_ptr, out_ptr,
    D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0)

    # x = relu(x + b0)  (bf16 add then relu, matching PyTorch elementwise semantics)
    xf = tl.cast(x, tl.float32) + tl.cast(b0, tl.float32)
    x_bf = tl.cast(xf, tl.bfloat16)
    x_bf = tl.maximum(x_bf, tl.cast(0.0, tl.bfloat16))

    # RMSNorm 1
    xf = tl.cast(x_bf, tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + EPS)
    y_bf = tl.cast(xf * inv, tl.bfloat16)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0)
    y_bf = tl.cast(tl.cast(y_bf, tl.float32) * tl.cast(w2, tl.float32), tl.bfloat16)

    # RMSNorm 2
    yf = tl.cast(y_bf, tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / D
    inv2 = 1.0 / tl.sqrt(ms2 + EPS)
    z_bf = tl.cast(yf * inv2, tl.bfloat16)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0)
    z_bf = tl.cast(tl.cast(z_bf, tl.float32) * tl.cast(w3, tl.float32), tl.bfloat16)

    tl.store(out_ptr + row * D + offs, z_bf, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.relu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        out = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            xc, self.b0, self.rms2_w, self.rms3_w, out,
            D=d, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
