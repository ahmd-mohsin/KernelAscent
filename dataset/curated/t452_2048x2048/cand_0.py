import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 452
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, W2, W3, G, B, Out,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.499  (bf16 elementwise op: compute fp32, round to bf16)
    x = (x * 1.499).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 1
    r = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = ((x * r).to(tl.bfloat16).to(tl.float32) * w2).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    r = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = ((x * r).to(tl.bfloat16).to(tl.float32) * w3).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 math, eps=1e-5)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b

    tl.store(Out + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(Mrows,)](
            x, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b, out,
            N, x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
