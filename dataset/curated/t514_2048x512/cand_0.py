import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 514
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_softmax_rms2_kernel(
    X, W2, W3, OUT,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 compute, cast to fp16 like torch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)

    # RMSNorm 1
    xf = p16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * r).to(tl.float16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    y16 = y16 * w2  # fp16 mul, matches eager

    # RMSNorm 2
    xf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    z16 = (xf * r).to(tl.float16)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    z16 = z16 * w3

    tl.store(OUT + row * N + offs, z16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_rms2_kernel[(rows,)](
            h, self.rms2_w, self.rms3_w, out,
            N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
