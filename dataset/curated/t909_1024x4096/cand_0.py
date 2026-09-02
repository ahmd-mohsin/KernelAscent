import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 909
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _ln_softmax_gelu_kernel(
    X, G, B, OUT,
    N, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch on half)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # round to fp16 between stages to match reference numerics
    y = y.to(tl.float16).to(tl.float32)

    # Softmax (fp32 math)
    y_masked = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    p = p.to(tl.float16).to(tl.float32)

    # GELU (exact, erf-based)
    out = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(OUT + row * stride_row + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (fp16, tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_softmax_gelu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, out,
            N, h.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
