import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 367
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_rms_softmax(X, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU in fp32, rounded to bf16 to match PyTorch output dtype
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g_bf16 = g.to(tl.bfloat16)
    gf = g_bf16.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    normed = (gf * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (normed * w).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_rms_softmax[(Mrows,)](
            h, self.rms2_w, out, N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
