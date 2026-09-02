import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 710
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_double_ln_kernel(
    X, Y, G0, B0, G1, B1,
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 accumulation, like PyTorch bf16 layer_norm) ----
    mean0 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(xc * xc, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + EPS)

    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    h = xc * rstd0 * g0 + b0
    # PyTorch materializes bf16 between the two layer norms -> round here
    h = h.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean1 = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    hc = tl.where(mask, h - mean1, 0.0)
    var1 = tl.sum(hc * hc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = hc * rstd1 * g1 + b1

    tl.store(Y + row * stride + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _rmsnorm_kernel(
    X, Y, W,
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)
    # match reference: normalize in fp32, round to bf16, then multiply by weight
    n = (x * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (n * w).to(tl.bfloat16)
    tl.store(Y + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape

        # Fused double LayerNorm (single pass over x instead of two kernels)
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_double_ln_kernel[(Mrows,)](
            x, y,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b,
            Dcols, Dcols,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )

        # cuBLAS GEMM (optimal for bf16 on A100 tensor cores)
        z = y @ self.W2

        # Fused RMSNorm
        Ncols = z.shape[-1]
        out = torch.empty_like(z)
        BLOCK2 = triton.next_power_of_2(Ncols)
        _rmsnorm_kernel[(Mrows,)](
            z, out, self.rms3_w,
            Ncols, Ncols,
            EPS=1e-6,
            BLOCK=BLOCK2,
            num_warps=8,
        )
        return out
