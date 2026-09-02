import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 881
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_softmax_rms_kernel(
    X, W, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load and scale (opmath in fp32, round back to fp16 like PyTorch)
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * 1.1029).to(tl.float16).to(tl.float32)

    # ---- softmax 1 (fp32 accum, fp16 output) ----
    xm = tl.where(mask, x, float('-inf'))
    mx = tl.max(xm, 0)
    e = tl.exp(xm - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- RMSNorm in fp32, cast to fp16, multiply by fp16 weight ----
    msq = tl.sum(tl.where(mask, x * x, 0.0), 0) / D
    r = 1.0 / tl.sqrt(msq + 1e-6)
    w = tl.load(W + offs, mask=mask, other=0.0)
    xh = (x * r).to(tl.float16) * w  # fp16 * fp16 multiply
    x = xh.to(tl.float32)

    # ---- softmax 2 ----
    xm = tl.where(mask, x, float('-inf'))
    mx = tl.max(xm, 0)
    e = tl.exp(xm - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- softmax 3 ----
    xm = tl.where(mask, x, float('-inf'))
    mx = tl.max(xm, 0)
    e = tl.exp(xm - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x * 1.1029
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x

        x = x.contiguous()
        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.view(-1, d)
        m = x2d.shape[0]

        y = torch.empty_like(x2d, dtype=torch.float16)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_softmax_rms_kernel[(m,)](
            x2d, self.rms2_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
