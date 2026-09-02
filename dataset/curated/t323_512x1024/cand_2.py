import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 323
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_rms_relu_bias_softmax_bias(
    X, W, B2, B4, Y,
    D: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- RMSNorm (fp32 accumulate, like reference) ----
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    inv = tl.math.rsqrt(ms + eps)
    xn = (x * inv).to(tl.float16)

    # ---- scale by weight (fp16 arithmetic, like reference) ----
    w = tl.load(W + offs, mask=mask, other=0.0)
    v = xn * w

    # ---- ReLU (fp16) ----
    zero = tl.zeros([BLOCK], dtype=tl.float16)
    v = tl.maximum(v, zero)

    # ---- + b2 (fp16) ----
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    v = v + b2

    # ---- softmax (fp32 internal math, fp16 output, like PyTorch half softmax) ----
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    mx = tl.max(vf, axis=0)
    e = tl.math.exp(vf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # ---- + b4 (fp16) ----
    b4 = tl.load(B4 + offs, mask=mask, other=0.0)
    out = p + b4

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            xx = torch.relu(xx)
            xx = xx + self.b2
            xx = torch.softmax(xx, dim=-1)
            return xx + self.b4

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_rms_relu_bias_softmax_bias[(n_rows,)](
            xc, self.rms0_w, self.b2, self.b4, y,
            D=d,
            eps=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
