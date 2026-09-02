import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 995
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_ln_gelu2_rms_relu(
    X, G, B, W, Y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch on fp16 inputs)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)  # cast to fp16 like reference

    # GELU (exact, erf-based) x2, with fp16 casts between ops
    SQRT1_2: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQRT1_2))
    y = y.to(tl.float16).to(tl.float32)
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQRT1_2))
    y = y.to(tl.float16).to(tl.float32)

    # RMSNorm (fp32 accumulation as in reference)
    y2 = tl.where(mask, y * y, 0.0)
    ms = tl.sum(y2, axis=0) / N
    rrms = tl.rsqrt(ms + 1e-6)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * rrms).to(tl.float16).to(tl.float32) * w
    y = y.to(tl.float16)

    # ReLU
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        x = torch.matmul(x, self.W0)
        x = x.contiguous()

        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)

        _fused_ln_gelu2_rms_relu[(Mrows,)](
            x, self.ln1_g, self.ln1_b, self.rms4_w, out,
            N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
