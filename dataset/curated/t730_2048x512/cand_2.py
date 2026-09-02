import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 730
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_double_ln_bias(
    X, OUT, G1, B1, G2, B2, B3,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First LayerNorm (fp32 accumulation, like PyTorch)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1
    # round to bf16 (output of first layer_norm) then promote back
    y1 = y1.to(tl.bfloat16).to(tl.float32)
    y1 = tl.where(mask, y1, 0.0)

    # Second LayerNorm
    mean2 = tl.sum(y1, axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = d2 * rstd2 * g2 + b2
    # round to bf16 (LN output), then bf16 add with b3 (fp32 compute, bf16 round)
    y2 = y2.to(tl.bfloat16).to(tl.float32)

    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y2 + b3).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_double_ln_bias[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.b3,
            N, h.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
