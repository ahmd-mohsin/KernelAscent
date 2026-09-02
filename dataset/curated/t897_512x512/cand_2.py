import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 897
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_relu_softmax_double_rms_kernel(
    X, Y, W2, W3,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- load + relu (fp32 compute) ----
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))

    # ---- softmax over the row (fp32 accumulation, like PyTorch) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    xb = sm.to(tl.bfloat16)          # round to bf16 (softmax output dtype)

    # ---- RMSNorm #1 ----
    xf = xb.to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / D
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = ((xf * r1).to(tl.bfloat16).to(tl.float32) * w2).to(tl.bfloat16)

    # ---- RMSNorm #2 ----
    xf2 = x1.to(tl.float32)
    ms2 = tl.sum(xf2 * xf2, axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((xf2 * r2).to(tl.bfloat16).to(tl.float32) * w3).to(tl.bfloat16)

    tl.store(Y + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_relu_softmax_double_rms_kernel[(n_rows,)](
            xc, y, self.rms2_w, self.rms3_w,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
