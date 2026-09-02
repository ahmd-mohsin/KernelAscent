import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 823
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_double_ln_kernel(
    X, G2, B2, G3, B3, Y,
    stride_x, stride_y,
    N,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load matmul output (fp16), apply 1.3509 scale with fp16 rounding ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * 1.3509
    x = x.to(tl.float16).to(tl.float32)  # emulate fp16 storage between ops

    # ---- LayerNorm 1 (fp32 stats, like PyTorch) ----
    mean1 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g2 + b2
    y1 = y1.to(tl.float16).to(tl.float32)  # emulate fp16 intermediate

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(tl.where(mask, y1, 0.0), axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)

    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    y2 = d2 * rstd2 * g3 + b3
    y2 = y2.to(tl.float16)

    # ---- final scale 1.4332 in fp16 (matches fp16 tensor * float scalar) ----
    scale = tl.full((), 1.4332, tl.float16)
    out = y2 * scale

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Tensor-core GEMM via cuBLAS
        h = torch.matmul(x, self.W0)

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)

        _fused_double_ln_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b, out,
            h.stride(0), out.stride(0),
            N,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
