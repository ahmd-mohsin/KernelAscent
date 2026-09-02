import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 826
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch's fp16 layer_norm internals)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    # multiply by scalar (opmath fp32), round to fp16 as reference does
    y16 = (y * SCALE).to(tl.float16)

    # RMSNorm on the fp16-rounded values, cast up to fp32
    xf = y16.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    z16 = (xf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z16.to(tl.float32) * w).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = x * 1.1102
            _xf = x.float()
            return (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_rms_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.rms2_w, out,
            x2.stride(0), out.stride(0),
            N=N,
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            SCALE=1.1102,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
