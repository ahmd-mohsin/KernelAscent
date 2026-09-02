import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 737
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_rms_kernel(
    x_ptr, w_ptr, b1_ptr, b2_ptr, out_ptr,
    D: tl.constexpr,
    stride_xm, stride_om,
    scale,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMS norm (fp32 accumulate)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + eps)

    # round to fp16 after normalization (matches .to(x.dtype))
    y = (xf * inv).to(tl.float16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    # each op: compute in fp32, round back to fp16 (PyTorch opmath semantics)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    y = (y.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    y = (y.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)
    y = (y.to(tl.float32) * scale).to(tl.float16)

    tl.store(out_ptr + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = y + self.b1
            y = y + self.b2
            return y * 1.2939

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 2048 else 8

        _fused_rms_kernel[(m,)](
            x2, self.rms0_w, self.b1, self.b2, out,
            d,
            x2.stride(0), out.stride(0),
            1.2939,
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
