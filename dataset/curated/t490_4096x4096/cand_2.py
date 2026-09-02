import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 490
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_gelu_ln_gelu_relu(
    X, G, B, Y,
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then rounded to fp16 to match
    # the reference's intermediate fp16 tensor
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)

    # LayerNorm statistics in fp32 (matches PyTorch CUDA half layernorm)
    mean = tl.sum(g1, axis=0) / N
    diff = tl.where(mask, g1 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    y = (g1 - mean) * rstd * gamma + beta
    y = y.to(tl.float16).to(tl.float32)

    # GELU + ReLU
    y2 = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y2 = tl.maximum(y2, 0.0)

    tl.store(Y + row * stride + offs, y2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)

        h2d = h.view(-1, h.shape[-1])
        rows, N = h2d.shape
        out = torch.empty_like(h2d)

        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_gelu_relu[(rows,)](
            h2d, self.ln2_g, self.ln2_b, out,
            N, h2d.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(h.shape)
