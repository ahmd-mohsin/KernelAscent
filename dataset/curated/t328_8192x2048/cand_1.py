import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 328
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_rms_softmax2_rms_relu(
    X, W0, W3, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- load input (fp16) ----
    x = tl.load(X + base + offs, mask=mask, other=0.0)

    # ---- RMSNorm 0 (compute in fp32, cast to fp16, multiply by fp16 weight) ----
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * r).to(tl.float16)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    xh = xh * w0  # fp16 * fp16 -> fp16 (matches PyTorch)

    # ---- Softmax 1 (fp32 accumulation, fp16 output; matches PyTorch half softmax) ----
    xf = tl.where(mask, xh.to(tl.float32), float('-inf'))
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    xh = (e / s).to(tl.float16)

    # ---- Softmax 2 ----
    xf = tl.where(mask, xh.to(tl.float32), float('-inf'))
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    xh = (e / s).to(tl.float16)

    # ---- RMSNorm 3 ----
    xf = xh.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * r).to(tl.float16)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    xh = xh * w3

    # ---- ReLU ----
    zero = tl.zeros(xh.shape, dtype=tl.float16)
    y = tl.maximum(xh, zero)

    tl.store(Y + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return torch.relu(x)

        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_rms_softmax2_rms_relu[(rows,)](
            x, self.rms0_w, self.rms3_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=8, num_stages=1,
        )
        return y
