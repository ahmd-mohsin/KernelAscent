import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 376
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_scale_ln_scale_relu(
    X, G, B, Y,
    N,
    eps,
    s1, s2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # replicate fp16 rounding of the elementwise scale before layer_norm
    x = (x * s1).to(tl.float16).to(tl.float32)

    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = d * rstd

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xn * g + b).to(tl.float16).to(tl.float32)   # LN output rounded to fp16
    y = (y * s2).to(tl.float16)                      # scale rounded to fp16
    y = tl.maximum(y, 0.0)                           # relu (idempotent, once is enough)

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (already optimal on A100)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_ln_scale_relu[(Mrows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, 1e-5, 1.1508, 1.2247,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
