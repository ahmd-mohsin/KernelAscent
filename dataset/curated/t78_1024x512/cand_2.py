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
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # --- LayerNorm 0 (fp32 math, like PyTorch bf16 layer_norm) ---
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + eps)

    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y0 = d0 * rstd0 * g0 + b0
    # round to bf16 (intermediate cast in reference), back to fp32
    y0 = y0.to(tl.bfloat16).to(tl.float32)

    # --- LayerNorm 1 ---
    mean1 = tl.sum(tl.where(mask, y0, 0.0), axis=0) / N
    d1 = tl.where(mask, y0 - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1

    tl.store(Y + row * N + cols, y1.to(tl.bfloat16), mask=mask)


@triton.jit
def _bias_scale_kernel(
    Y, B, Out,
    n_elem, N, scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem

    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + (offs % N), mask=mask, other=0.0).to(tl.float32)

    # match: (y + b) -> bf16 round, then * scale -> bf16 round
    t = (y + b).to(tl.bfloat16).to(tl.float32)
    out = (t * scale).to(tl.bfloat16)

    tl.store(Out + offs, out, mask=mask)


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

        # Fused double layernorm
        x_ln = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _double_ln_kernel[(m,)](
            x, self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b, x_ln,
            n, 1e-5, BLOCK=BLOCK, num_warps=4,
        )

        # Matmul via cuBLAS (bf16 tensor cores)
        y = torch.matmul(x_ln, self.W2)

        # Fused bias add + scale
        out = torch.empty_like(y)
        n_out = y.shape[-1]
        n_elem = y.numel()
        BLOCK_E = 1024
        grid = (triton.cdiv(n_elem, BLOCK_E),)
        _bias_scale_kernel[grid](
            y, self.b3, out,
            n_elem, n_out, 1.3616, BLOCK=BLOCK_E, num_warps=4,
        )
        return out
