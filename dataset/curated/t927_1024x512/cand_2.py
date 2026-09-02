import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 927
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, OUT,
    G1, B1, G2, B2, RW,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    x = x16.to(tl.float32)

    # ---- LayerNorm 1 (fp32 compute, fp16 round) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g1 + b1
    y = y.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 (fp32 compute, fp16 round) ----
    mean2 = tl.sum(y, axis=0) / N
    yc = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = yc * rstd2 * g2 + b2
    z16 = z.to(tl.float16)

    # ---- scale in fp16 (rounds like reference) ----
    scale = tl.full((), 1.1137, tl.float16)
    z16 = z16 * scale

    # ---- RMSNorm (fp32 compute, fp16 round, fp16 weight mul) ----
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    ms = tl.sum(zf * zf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    o16 = (zf * inv).to(tl.float16)
    rw = tl.load(RW + cols, mask=mask, other=0.0)
    o16 = o16 * rw

    tl.store(OUT + row * stride_o + cols, o16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS tensor-core GEMM
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_kernel[(Mrows,)](
            x, out,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.rms4_w,
            x.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
