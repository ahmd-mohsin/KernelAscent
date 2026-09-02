import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 649
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_rms_softmax_kernel(
    X, W1, W3, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm 1 (match PyTorch: fp32 compute, cast to fp16, fp16 multiply by weight)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    t = (xf * r).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    h = t * w1  # fp16 arithmetic

    # Softmax 1 (fp32 accumulation, fp16 output)
    hf = tl.where(mask, h.to(tl.float32), float('-inf'))
    m1 = tl.max(hf, axis=0)
    e1 = tl.exp(hf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16)

    # RMSNorm 2
    pf = p1.to(tl.float32)
    ms2 = tl.sum(pf * pf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    t2 = (pf * r2).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    h2 = t2 * w3  # fp16 arithmetic

    # Softmax 2
    hf2 = tl.where(mask, h2.to(tl.float32), float('-inf'))
    m2 = tl.max(hf2, axis=0)
    e2 = tl.exp(hf2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.float16)

    tl.store(Out + row * stride_om + cols, p2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_softmax_kernel[(m,)](
            x, self.rms1_w, self.rms3_w, out,
            x.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
