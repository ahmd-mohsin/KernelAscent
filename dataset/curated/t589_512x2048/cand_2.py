import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 589
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _ln_bias_relu_gelu2_kernel(
    X_ptr, Y_ptr, G_ptr, B_ptr, B2_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, biased variance) -> round to bf16
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b2 (rounded to bf16 as in eager)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b2).to(tl.bfloat16).to(tl.float32)

    # ReLU
    y = tl.maximum(y, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # GELU (exact, erf) twice, with bf16 rounding between ops
    y = (y * 0.5 * (1.0 + tl.math.erf(y * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    y = y * 0.5 * (1.0 + tl.math.erf(y * INV_SQRT2))

    tl.store(Y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _ln_bias_relu_gelu2_kernel[(M_,)](
            h, out, self.ln1_g, self.ln1_b, self.b2,
            N_, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
