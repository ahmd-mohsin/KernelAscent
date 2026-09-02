import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 481
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_rms_relu_softmax(
    X, W, OUT,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu (bf16)
    x = tl.maximum(x, 0.0)

    # rmsnorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    # round normalized value to bf16, then multiply by weight (bf16 elementwise
    # mul on CUDA computes in fp32 and rounds back to bf16)
    xn = (xf * r).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    t = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # relu
    t = tl.maximum(t, 0.0)

    # softmax: fp32 accumulation, bf16 output (matches torch CUDA softmax)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    mval = tl.max(tf, axis=0)
    e = tl.exp(tf - mval)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (bf16 tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_relu_softmax[(m,)](
            y, self.rms2_w, out,
            n, y.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
