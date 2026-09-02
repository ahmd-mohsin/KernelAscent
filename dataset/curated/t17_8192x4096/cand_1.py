import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 17
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_rms_softmax_ln(
    X, B1, RW, G, B, Y,
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)

    # bias add (rounded to bf16, matching reference)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm (stats in fp32, round to bf16, then weight mul rounded to bf16)
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    rw = tl.load(RW + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * rw).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, output rounded to bf16
    xm = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.exp(x - xm)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, p, 0.0), axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (p - mean) * inv * g + b

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_ln[(rows,)](
            x, self.b1, self.rms2_w, self.ln4_g, self.ln4_b, y,
            x.stride(0), y.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
