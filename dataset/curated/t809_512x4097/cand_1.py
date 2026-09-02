import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 809
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_act_ln2_kernel(
    X_ptr, G4_ptr, B4_ptr, G5_ptr, B5_ptr, Y_ptr,
    N, EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # relu (exact in bf16)
    x = tl.maximum(x, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # gelu #1 (compute fp32, round to bf16 like PyTorch elementwise op)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)
    # gelu #2
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (fp32 math, output rounded to bf16 like PyTorch)
    x = tl.where(mask, x, 0.0)
    mean1 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(xc * xc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g4 = tl.load(G4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd1 * g4 + b4
    y = y.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    y = tl.where(mask, y, 0.0)
    mean2 = tl.sum(y, axis=0) / N
    yc = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g5 = tl.load(G5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = yc * rstd2 * g5 + b5

    tl.store(Y_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (identical to reference)
        if not h.is_cuda:
            h = torch.relu(h)
            h = F.gelu(h)
            h = F.gelu(h)
            h = F.layer_norm(h, (h.shape[-1],), self.ln4_g, self.ln4_b)
            h = F.layer_norm(h, (h.shape[-1],), self.ln5_g, self.ln5_b)
            return h

        h = h.contiguous()
        rows, N = h.shape[0], h.shape[1]
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_act_ln2_kernel[(rows,)](
            h, self.ln4_g, self.ln4_b, self.ln5_g, self.ln5_b, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
