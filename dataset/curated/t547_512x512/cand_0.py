import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 547
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_epilogue(Y, B2, LG, LB, B4, RW, OUT,
                    N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    # load matmul output (bf16 -> fp32)
    y = tl.load(Y + base + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32, rounded to bf16 (matches F.gelu on bf16 tensor)
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # + b2 (bf16 add: fp32 compute, bf16 round)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    t = (g + b2).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 stats, biased variance, eps=1e-5)
    mean = tl.sum(tl.where(mask, t, 0.0), axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    lg = tl.load(LG + offs, mask=mask, other=0.0).to(tl.float32)
    lb = tl.load(LB + offs, mask=mask, other=0.0).to(tl.float32)
    o = ((t - mean) * rstd * lg + lb).to(tl.bfloat16).to(tl.float32)

    # + b4
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    u = (o + b4).to(tl.bfloat16).to(tl.float32)

    # RMSNorm: fp32 mean-of-squares, eps=1e-6, round to bf16, then * weight
    ms = tl.sum(tl.where(mask, u * u, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    v = (u * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(RW + offs, mask=mask, other=0.0).to(tl.float32)
    out = (v * w).to(tl.bfloat16)

    tl.store(OUT + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (bf16 with fp32 accumulate), same as reference
        y = x @ self.W0

        orig_shape = y.shape
        y2 = y.reshape(-1, orig_shape[-1]).contiguous()
        rows, N = y2.shape
        out = torch.empty_like(y2)

        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue[(rows,)](
            y2, self.b2, self.ln3_g, self.ln3_b, self.b4, self.rms5_w, out,
            N=N, BLOCK=BLOCK, num_warps=8,
        )
        return out.reshape(orig_shape)
