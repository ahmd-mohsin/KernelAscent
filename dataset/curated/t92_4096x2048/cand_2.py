import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 92
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, OUT,
    G1, B1, B2P, G3, B3, RW,
    stride,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, fp16 output like PyTorch) ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)

    g1 = tl.load(G1 + cols, mask=mask).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
    y = (d1 * rstd1) * g1 + b1
    y16 = y.to(tl.float16)

    # ---- add bias b2 (fp16 add, matching reference) ----
    b2 = tl.load(B2P + cols, mask=mask)
    y16 = y16 + b2

    # ---- LayerNorm 3 ----
    x2 = y16.to(tl.float32)
    mean2 = tl.sum(tl.where(mask, x2, 0.0), axis=0) / N
    d2 = tl.where(mask, x2 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)

    g3 = tl.load(G3 + cols, mask=mask).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask).to(tl.float32)
    z = (d2 * rstd2) * g3 + b3
    z16 = z.to(tl.float16)

    # ---- RMSNorm (fp32 like reference, cast to fp16, then fp16 * weight) ----
    zf = z16.to(tl.float32)
    ms = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + 1e-6)
    r16 = (zf * rinv).to(tl.float16)

    rw = tl.load(RW + cols, mask=mask)
    out = r16 * rw

    # ---- ReLU ----
    out = tl.where(out > 0, out, 0.0)

    tl.store(OUT + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Tensor-core GEMM
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_norms_kernel[(m,)](
            h, out,
            self.ln1_g, self.ln1_b, self.b2, self.ln3_g, self.ln3_b, self.rms4_w,
            h.stride(0),
            N=n,
            BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
