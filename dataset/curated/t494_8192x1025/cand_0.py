import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 494
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_norms_gelu_kernel(
    X_ptr, W1_ptr, W2_ptr, G_ptr, B_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 1 ----
    ms1 = tl.sum(x * x, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    xb = (x * r1).to(tl.bfloat16)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xb = (xb.to(tl.float32) * w1).to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    xf = xb.to(tl.float32)
    ms2 = tl.sum(xf * xf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    xb = (xf * r2).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xb = (xb.to(tl.float32) * w2).to(tl.bfloat16)

    # ---- LayerNorm ----
    xf = xb.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    yb = (diff * rstd * g + b).to(tl.bfloat16)

    # ---- GELU (erf, exact) ----
    yf = yb.to(tl.float32)
    out = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already optimal on A100 tensor cores)
        x = x @ self.W0

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.reshape(-1, N).contiguous()
        rows = x2d.shape[0]

        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_norms_gelu_kernel[(rows,)](
            x2d, self.rms1_w, self.rms2_w, self.ln3_g, self.ln3_b, y,
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
