import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 458
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_relu_ln_bias_scale_softmax(
    Y_ptr, OUT_ptr, G_ptr, B_ptr, B3_ptr,
    stride_ym, stride_om,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_ym + cols, mask=mask, other=0.0).to(tl.float32)
    # relu
    y = tl.maximum(y, 0.0)

    # layernorm (fp32 stats)
    mean = tl.sum(y, axis=0) / N
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    xn = diff * rstd * g + b
    # emulate bf16 rounding between ops (matches reference sequential ops)
    xn = xn.to(tl.bfloat16).to(tl.float32)
    xn = (xn + b3).to(tl.bfloat16).to(tl.float32)
    xn = (xn * scale).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32)
    xn = tl.where(mask, xn, float('-inf'))
    m = tl.max(xn, axis=0)
    e = tl.exp(xn - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_ln_bias_scale_softmax[(Mrows,)](
            y, out, self.ln2_g, self.ln2_b, self.b3,
            y.stride(0), out.stride(0),
            N, 1e-5, 1.1477,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
