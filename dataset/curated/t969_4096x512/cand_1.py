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
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in float32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    normed = xf * inv
    # cast to input dtype (bf16), then multiply by weight (fp32 math, round to bf16)
    normed_bf = normed.to(X.dtype.element_ty)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (normed_bf.to(tl.float32) * w.to(tl.float32)).to(X.dtype.element_ty)

    # softmax in float32
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, float('-inf'))
    m = tl.max(hf, axis=0)
    e = tl.exp(hf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(Y.dtype.element_ty)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _rmsnorm_softmax_kernel[(Mrows,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
