import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 227
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_ln_gelu_ln_gelu(
    X, Y, G1, B1, G3, B3,
    N, stride_x, stride_y,
    EPS, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 stats, like PyTorch half layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g1 + b1
    # round to fp16 (PyTorch materializes fp16 tensor between ops)
    y = y.to(tl.float16).to(tl.float32)

    # ---- GELU (exact erf, fp32 opmath) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    ym = tl.where(mask, y, 0.0)
    mean2 = tl.sum(ym, axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g3 + b3
    z = z.to(tl.float16).to(tl.float32)

    # ---- scale ----
    z = z * SCALE
    z = z.to(tl.float16).to(tl.float32)

    # ---- GELU ----
    z = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, z.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_gelu_ln_gelu[(rows,)](
            h2, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            N, h2.stride(0), out.stride(0),
            1e-5, 1.4046,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out.reshape(orig_shape)
