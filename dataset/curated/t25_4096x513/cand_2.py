import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 25
M, D, DT = 4096, 513, torch.bfloat16


@triton.jit
def _fused_rms_softmax_kernel(
    X, W, OUT,
    N,  # row length (2048)
    stride_xm, stride_om,
    S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    xf = x.to(tl.float32)

    # RMS norm (mean over N)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    # (_xf * rsqrt).to(bf16)
    y = (xf * inv).to(tl.bfloat16)

    # * rms1_w  (bf16 elementwise mul -> computed in fp32, rounded to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0)  # bf16
    z = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # * 1.2879 (round to bf16), * 1.2562 (round to bf16)
    z = (z.to(tl.float32) * S1).to(tl.bfloat16)
    z = (z.to(tl.float32) * S2).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch's bf16 softmax which upcasts internally)
    zf = tl.where(mask, z.to(tl.float32), float("-inf"))
    m = tl.max(zf, axis=0)
    e = tl.exp(zf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16)

    # * 1.2744 (round to bf16)
    out = (p.to(tl.float32) * S3).to(tl.bfloat16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_softmax_kernel[(m,)](
            x, self.rms1_w, out,
            n,
            x.stride(0), out.stride(0),
            1.2879, 1.2562, 1.2744,
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
