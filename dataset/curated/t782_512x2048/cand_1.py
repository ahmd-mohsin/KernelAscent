import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 782
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_act_rms_softmax(X, W, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (bf16) and upcast
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # relu (exact on bf16 values)
    r = tl.maximum(x, 0.0)

    # exact gelu computed in fp32, rounded to bf16 (matches F.gelu opmath behavior)
    g = 0.5 * r * (1.0 + tl.math.erf(r * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)
    gf = g_bf.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    y_bf = (gf * inv).to(tl.bfloat16)

    # multiply by weight in bf16 (matches reference: bf16 tensor * bf16 weight)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y_bf = y_bf * w

    # softmax with fp32 accumulation, output bf16
    yf = y_bf.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    mx = tl.max(yf, axis=0)
    e = tl.exp(yf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_act_rms_softmax[(m,)](
            x, self.rms3_w, y, N=n, BLOCK=BLOCK, num_warps=8
        )
        return y @ self.W5
