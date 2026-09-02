import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 339
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_gelu_scale_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu(x) computed in fp32, rounded to fp16 (matches PyTorch half gelu)
    xf = x.to(tl.float32)
    g1 = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g1 = g1.to(tl.float16)

    # add bias in fp16
    y = g1 + b

    # second gelu
    yf = y.to(tl.float32)
    g2 = yf * 0.5 * (1.0 + tl.math.erf(yf * INV_SQRT2))
    g2 = g2.to(tl.float16)

    # scalar multiply (fp32 compute, cast to fp16)
    z = (g2.to(tl.float32) * 1.437).to(tl.float16)

    # softmax in fp32, output fp16
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    m = tl.max(zf, axis=0)
    e = tl.exp(zf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_bias_gelu_scale_softmax[(Mrows,)](
            h, self.b2, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
