import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 680
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_rms_rms_kernel(
    X, W1, W3, OUT,
    D: tl.constexpr,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- softmax (fp32 accumulation, matching PyTorch half softmax) ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)  # softmax output cast to fp16

    # ---- rmsnorm 1 ----
    yf = y.to(tl.float32)
    ms1 = tl.sum(yf * yf, axis=0) / D
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    y1 = (yf * r1).to(tl.float16)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    # half*half mul is computed in fp32 (opmath) then cast back to fp16
    y1 = (y1.to(tl.float32) * w1.to(tl.float32)).to(tl.float16)
    # scalar mul: opmath fp32 then cast back to fp16
    y1 = (y1.to(tl.float32) * 1.4345).to(tl.float16)

    # ---- rmsnorm 2 ----
    zf = y1.to(tl.float32)
    ms2 = tl.sum(zf * zf, axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    z = (zf * r2).to(tl.float16)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    out = (z.to(tl.float32) * w3.to(tl.float32)).to(tl.float16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference fallback
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = x * 1.4345
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_softmax_rms_rms_kernel[(rows,)](
            x2, self.rms1_w, self.rms3_w, out,
            d,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
