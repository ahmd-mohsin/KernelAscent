import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 514
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _softmax_double_rms_kernel(
    X, W2, W3, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, fp16 output — matches PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # RMSNorm 1
    xf = p.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    v = (xf * r).to(tl.float16) * w2  # fp16 multiply, matching reference

    # RMSNorm 2
    vf = v.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, vf * vf, 0.0), axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    out = (vf * r2).to(tl.float16) * w3

    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _softmax_double_rms_kernel[(rows,)](
            y, self.rms2_w, self.rms3_w, out,
            y.stride(0), out.stride(0),
            N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
