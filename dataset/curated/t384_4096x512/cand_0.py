import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 384
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_gelu_relu_rms2_kernel(
    X, W3, W4, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # exact (erf) GELU computed in fp32, rounded to fp16 (matches PyTorch opmath)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)
    # relu on fp16
    r16 = tl.maximum(g16, tl.zeros_like(g16))

    # first RMSNorm
    f1 = r16.to(tl.float32)
    ms1 = tl.sum(tl.where(mask, f1 * f1, 0.0), axis=0) / N
    inv1 = tl.math.rsqrt(ms1 + 1e-6)
    n1_16 = (f1 * inv1).to(tl.float16)

    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    y1_16 = (n1_16.to(tl.float32) * w3).to(tl.float16)

    # second RMSNorm
    f2 = y1_16.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, f2 * f2, 0.0), axis=0) / N
    inv2 = tl.math.rsqrt(ms2 + 1e-6)
    n2_16 = (f2 * inv2).to(tl.float16)

    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.float32)
    y2_16 = (n2_16.to(tl.float32) * w4).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y2_16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_relu_rms2_kernel[(Mrows,)](
            h, self.rms3_w, self.rms4_w, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
