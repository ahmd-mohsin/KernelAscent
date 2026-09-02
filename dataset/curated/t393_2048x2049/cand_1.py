import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 393
M, D, DT = 2048, 2049, torch.float16


@triton.jit
def _fused_gelu_bias_gelu_softmax(X, B, Out,
                                  N, stride_xm, stride_om,
                                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), computed in fp32 then rounded to fp16 (matches PyTorch opmath)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b
    y = y.to(tl.float16).to(tl.float32)

    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    # softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 via cuBLAS tensor cores
        h = x @ self.W0  # (M, 1024) fp16

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_bias_gelu_softmax[(Mrows,)](
            h, self.b2, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2 via cuBLAS
        return out @ self.W5
