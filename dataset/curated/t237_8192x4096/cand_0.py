import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 237
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_ln_gelu_ln_relu(
    X, Y, G1, B1, G3, B3,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 math, output rounded to fp16 like PyTorch)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    h = ((x - mean) * rstd * g1 + b1).to(tl.float16)

    # GELU (exact erf, computed in fp32, rounded back to fp16)
    hf = h.to(tl.float32)
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    gelu = hf * 0.5 * (1.0 + tl.math.erf(hf * inv_sqrt2))
    gelu = gelu.to(tl.float16)

    # LayerNorm 3
    z = gelu.to(tl.float32)
    z = tl.where(mask, z, 0.0)
    mean2 = tl.sum(z, axis=0) / N
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    o = ((z - mean2) * rstd2 * g3 + b3).to(tl.float16)

    # scale (fp32 opmath, rounded to fp16) then ReLU
    o = (o.to(tl.float32) * 1.4996).to(tl.float16)
    o = tl.maximum(o, tl.zeros_like(o))

    tl.store(Y + row * stride_y + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (fp32 accumulate)
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_gelu_ln_relu[(rows,)](
            h, y,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            h.stride(0), y.stride(0),
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
