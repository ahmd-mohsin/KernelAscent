import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 28
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_scale_rms_softmax(
    X, W, Out,
    stride_xm, stride_om,
    N, eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # fp16

    # x = x * 1.1067  (compute in fp32, store back to fp16, matching PyTorch opmath)
    xf = x.to(tl.float32) * scale
    x16 = xf.to(tl.float16)

    # RMSNorm in fp32
    xf2 = x16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf2 * xf2, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    xn16 = (xf2 * inv).to(tl.float16)

    # multiply by weight (fp16 tensors, PyTorch computes in fp32 then casts to fp16)
    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    y16 = (xn16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # softmax (fp16 input, fp32 accumulation, fp16 output)
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_scale_rms_softmax[(m,)](
            x, self.rms2_w, out,
            x.stride(0), out.stride(0),
            n, 1e-6, 1.1067,
            BLOCK_N=BLOCK_N,
            num_warps=16,
        )
        return out
