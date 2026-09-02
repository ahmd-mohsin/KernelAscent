import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 252
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu_bias_ln_kernel(
    X_ptr, B1_ptr, G_ptr, B_ptr, Y_ptr,
    N_COLS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    x = tl.load(X_ptr + row * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))

    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    v = g + b1

    mean = tl.sum(v, axis=0) / N_COLS
    diff = tl.where(mask, v - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N_COLS
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = (v - mean) * rstd * gamma + beta
    tl.store(Y_ptr + row * N_COLS + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x + self.b1
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_gelu_bias_ln_kernel[(n_rows,)](
            x2, self.b1, self.ln2_g, self.ln2_b, y,
            N_COLS=n_cols, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
