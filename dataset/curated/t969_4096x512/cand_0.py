import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 969
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _rmsnorm_softmax_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in float32
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    normed = xf * inv

    # cast to bf16 (matches .to(x.dtype)), then multiply by weight -> bf16
    normed_bf = normed.to(tl.bfloat16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    v_bf = (normed_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax computed in float32 from bf16 values (matches PyTorch bf16 softmax)
    v = v_bf.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))
    vmax = tl.max(v, axis=0)
    e = tl.exp(v - vmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _rmsnorm_softmax_kernel[(m,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
