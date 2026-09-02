import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 552
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _rms_softmax_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    normed = xf * inv

    # cast to bf16 (matches .to(x.dtype)), then multiply by weight
    # PyTorch bf16 mul: computed in fp32, rounded back to bf16
    normed_bf = normed.to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    prod_bf = (normed_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax: bf16 input, fp32 accumulation (matches PyTorch)
    z = prod_bf.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        _rms_softmax_kernel[(m,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            n, BLOCK=BLOCK, EPS=1e-6,
            num_warps=4,
        )
        return y.view(orig_shape)
