import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 540
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_rms_bias_gelu_softmax(
    X, W, B1, B2, Out,
    D: tl.constexpr,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 math, matches _xf.pow(2).mean(-1) then rsqrt)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)  # cast back to input dtype (bf16)

    # * rms0_w  (bf16 op == fp32 op then round-to-bf16, since bf16 values are exact fp32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn.to(tl.float32) * w).to(tl.bfloat16)

    # + b1
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) + b1).to(tl.bfloat16)

    # + b2
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) + b2).to(tl.bfloat16)

    # exact GELU (erf), computed in fp32 like PyTorch's CUDA kernel, rounded to bf16
    yf = y.to(tl.float32)
    g = (yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))).to(tl.bfloat16)

    # softmax in fp32 accumulation (matches PyTorch bf16 softmax)
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, float("-inf"))
    mx = tl.max(gf, axis=0)
    e = tl.exp(gf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_rms_bias_gelu_softmax[(m,)](
            x2, self.rms0_w, self.b1, self.b2, out,
            d,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
