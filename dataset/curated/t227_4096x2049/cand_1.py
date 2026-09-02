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
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 math, like PyTorch half layer_norm)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x - mean) * rstd * g1 + b1
    # round to fp16 (op boundary)
    x = x.to(tl.float16).to(tl.float32)

    # GELU (exact, erf), fp32 opmath then round to fp16
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(diff2 * diff2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x - mean2) * rstd2 * g3 + b3
    x = x.to(tl.float16).to(tl.float32)

    # scale (fp32 opmath, round)
    x = x * SCALE
    x = x.to(tl.float16).to(tl.float32)

    # GELU
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, x.to(tl.float16), mask=mask)


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
        if not x.is_cuda:
            x = x @ self.W0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = x * 1.4046
            x = F.gelu(x)
            return x

        h = x @ self.W0  # cuBLAS tensor-core GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _fused_ln_gelu_ln_gelu[(Mrows,)](
            h, y, self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            N, h.stride(0), y.stride(0),
            SCALE=1.4046, EPS=1e-5, BLOCK=512,
            num_warps=4,
        )
        return y
