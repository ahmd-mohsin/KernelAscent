import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 714
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_bias_relu_ln_rms(
    Y, B1, G, Bt, RW, OUT,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)

    # bias add + relu in fp16 (matches reference dtype semantics)
    x = y + b1
    zero = tl.zeros(x.shape, dtype=x.dtype)
    x = tl.maximum(x, zero)

    # layernorm in fp32
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(Bt + cols, mask=mask, other=0.0).to(tl.float32)
    xn = diff * inv * g + bt
    xh = xn.to(tl.float16)

    # rmsnorm: fp16 -> fp32, normalize, cast fp16, multiply weight in fp16
    xf2 = xh.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf2 * xf2, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    out16 = (xf2 * r).to(tl.float16)
    rw = tl.load(RW + cols, mask=mask, other=0.0)
    out = out16 * rw

    tl.store(OUT + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_relu_ln_rms[(m,)](
            y, self.b1, self.ln3_g, self.ln3_b, self.rms4_w, out,
            n, y.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return out
