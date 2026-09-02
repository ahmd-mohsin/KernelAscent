import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 137
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_rms_gelu_softmax(X, W, Out, N, stride_x, stride_o, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean over N valid elements)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w  # fp16 multiply, matching reference

    # exact GELU (erf-based) computed in fp32 (matches PyTorch opmath on half)
    v = y.to(tl.float32)
    g = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # softmax over the row (fp32 accumulation, matching PyTorch half softmax)
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    m = tl.max(gf, axis=0)
    e = tl.exp(gf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # (M, 512) fp16, tensor-core GEMM
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_softmax[(Mrows,)](
            x, self.rms1_w, out, N,
            x.stride(0), out.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
