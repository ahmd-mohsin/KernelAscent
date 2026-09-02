import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 78
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _double_ln_kernel(
    X, G0, B0, G1, B1, Y,
    N: tl.constexpr, eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    x = tl.load(X + row * N + cols).to(tl.float32)

    # First LayerNorm (fp32 math, like PyTorch's opmath for bf16)
    mean0 = tl.sum(x, axis=0) / N
    d0 = x - mean0
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + eps)
    g0 = tl.load(G0 + cols).to(tl.float32)
    b0 = tl.load(B0 + cols).to(tl.float32)
    y0 = d0 * rstd0 * g0 + b0
    # emulate the intermediate bf16 rounding between the two LayerNorms
    y0 = y0.to(tl.bfloat16).to(tl.float32)

    # Second LayerNorm
    mean1 = tl.sum(y0, axis=0) / N
    d1 = y0 - mean1
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + cols).to(tl.float32)
    b1 = tl.load(B1 + cols).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1

    tl.store(Y + row * N + cols, y1.to(tl.bfloat16))


@triton.jit
def _bias_scale_kernel(
    Y, B, n_elem, N: tl.constexpr, scale, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + (offs % N), mask=mask, other=0.0).to(tl.float32)
    # match reference: (x + b3) rounded to bf16, then * 1.3616, rounded to bf16
    s = (y + b).to(tl.bfloat16).to(tl.float32)
    out = (s * scale).to(tl.bfloat16)
    tl.store(Y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape

        # Fused double LayerNorm (single pass over rows)
        ln_out = torch.empty_like(x)
        _double_ln_kernel[(m,)](
            x, self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b, ln_out,
            N=n, eps=1e-5,
            num_warps=4,
        )

        # cuBLAS matmul (bf16, fp32 accumulate) — fastest path on A100
        out = torch.mm(ln_out, self.W2)

        # Fused bias-add + scale epilogue
        n_out = out.shape[1]
        n_elem = out.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elem, BLOCK),)
        _bias_scale_kernel[grid](
            out, self.b3, n_elem, N=n_out, scale=1.3616, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
