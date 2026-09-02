import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 171
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_rms_relu_rms(
    x_ptr, w2_ptr, w4_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- softmax (fp32 accumulation, fp16 output like PyTorch) ----
    x = tl.load(x_ptr + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    p16 = (e / s).to(tl.float16)

    # ---- scale by 1.0869 (opmath float, output fp16) ----
    xf = p16.to(tl.float32) * 1.0869
    x16 = xf.to(tl.float16)

    # ---- RMSNorm 1 ----
    xf = x16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    y16 = (xf * inv).to(tl.float16)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z16 = (y16.to(tl.float32) * w2).to(tl.float16)

    # ---- ReLU ----
    z16 = tl.maximum(z16, tl.zeros_like(z16))

    # ---- RMSNorm 2 ----
    zf = z16.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0) / D
    inv2 = tl.math.rsqrt(ms2 + 1e-6)
    y2 = (zf * inv2).to(tl.float16)
    w4 = tl.load(w4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out16 = (y2.to(tl.float32) * w4).to(tl.float16)

    tl.store(out_ptr + base + offs, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference implementation)
            x = torch.softmax(x, dim=-1)
            x = x * 1.0869
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_rms_relu_rms[(n_rows,)](
            x2d, self.rms2_w, self.rms4_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
