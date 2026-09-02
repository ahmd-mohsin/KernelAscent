import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 205
M, D, DT = 4096, 4096, torch.bfloat16

INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_kernel(X, G, B, B4, Y, N_COLS: tl.constexpr, EPS: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    base = row * N_COLS
    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)

    # relu (exact in bf16, no rounding change)
    x = tl.maximum(x, 0.0)

    # gelu #1 (exact erf variant, computed in fp32, rounded to bf16 like torch)
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 accumulation, like PyTorch)
    mean = tl.sum(x, axis=0) / N_COLS
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N_COLS
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # round to bf16 (layer_norm output), then add b4 (fp32 intermediate)
    y = y.to(tl.bfloat16).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b4

    tl.store(Y + base + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.relu(x)
            y = F.gelu(y)
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return y + self.b4

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(n_rows,)](
            x2, self.ln3_g, self.ln3_b, self.b4, out,
            N_COLS=n_cols, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
