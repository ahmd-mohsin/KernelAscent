import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 923
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _rms_gelu_softmax_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean of squares in fp32, matches reference)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    # cast normalized value to fp16, then multiply by fp16 weight (matches ref order)
    xn = (xf * r).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = xn * w  # fp16 multiply

    # exact GELU: compute in fp32, cast back to fp16 (matches PyTorch half gelu)
    yf = y.to(tl.float32)
    g = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    gh = g.to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax internals)
    gf = gh.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    mx = tl.max(gf, axis=0)
    e = tl.exp(gf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_ym + offs, out, mask=mask)


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape

        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_gelu_softmax_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )

        z = y @ self.W4
        z = z.contiguous()
        out = torch.empty_like(z)
        nel = z.numel()
        BLOCK_E = 1024
        grid = (triton.cdiv(nel, BLOCK_E),)
        _gelu_kernel[grid](z, out, nel, BLOCK=BLOCK_E, num_warps=4)
        return out
