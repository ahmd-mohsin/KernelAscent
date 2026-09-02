import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 174
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_ln_relu_add_ln(
    Y, OUT,
    G1, B1, B3, G2, B2,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 math, matching PyTorch opmath)
    mean1 = tl.sum(y, axis=0) / N
    d1 = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d1 * rstd1 * g1 + b1
    # round to bf16 (output of layer_norm), then relu (exact on bf16)
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # add b3 in bf16 semantics: fp32 add, round once to bf16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b3).to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    xm = tl.where(mask, x, 0.0)
    mean2 = tl.sum(xm, axis=0) / N
    d2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g2 + b2

    tl.store(OUT + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.matmul(x, self.W0)  # tensor-core GEMM, bf16
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_relu_add_ln[(Mrows,)](
            y, out,
            self.ln1_g, self.ln1_b, self.b3, self.ln4_g, self.ln4_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
