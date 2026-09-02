import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 69
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_rms_softmax_gelu(X, W, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    x = tl.load(ptr).to(tl.float32)

    # GELU (exact, erf-based), computed in fp32 then rounded to fp16 like PyTorch
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g16 = g.to(tl.float16)

    # RMSNorm: stats in fp32 on the fp16 values (matches x.float())
    gf = g16.to(tl.float32)
    ms = tl.sum(gf * gf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    h16 = (gf * r).to(tl.float16)

    # scale by weight in fp16 (matches fp16 * fp16)
    w = tl.load(W + offs)
    h16 = h16 * w

    # softmax in fp32, output rounded to fp16
    hf = h16.to(tl.float32)
    mx = tl.max(hf, axis=0)
    e = tl.exp(hf - mx)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)

    # final GELU in fp32 on fp16 softmax output
    pf = p16.to(tl.float32)
    out = 0.5 * pf * (1.0 + tl.math.erf(pf * INV_SQRT2))

    tl.store(Y + row * N + offs, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        _fused_gelu_rms_softmax_gelu[(rows,)](
            h, self.rms2_w, y, N,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y
