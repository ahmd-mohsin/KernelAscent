import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 521
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b1_ptr, w_ptr, out_ptr,
    stride_xm, stride_om,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0)

    # x * 1.405 (bf16 rounding to match torch)
    xf = x.to(tl.float32) * 1.405
    x = xf.to(tl.bfloat16)

    # x + b1 (bf16)
    xf = x.to(tl.float32) + b1.to(tl.float32)
    x = xf.to(tl.bfloat16)

    # relu (exact in bf16)
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)

    # softmax in fp32, output rounded to bf16 (matches torch CUDA softmax)
    xf_masked = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf_masked, axis=0)
    e = tl.exp(xf_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    sm_bf16 = sm.to(tl.bfloat16)

    # RMS norm in fp32
    sf = sm_bf16.to(tl.float32)
    ms = tl.sum(sf * sf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    normed = (sf * inv).to(tl.bfloat16)

    # multiply by weight (bf16 * bf16 -> bf16, computed in fp32 then rounded)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_om + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.405
            x = x + self.b1
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.b1, self.rms4_w, out,
            x.stride(0), out.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
