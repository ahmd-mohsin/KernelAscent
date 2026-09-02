import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 567
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_rms_relu_ln_gelu2(X, RW, G, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X + row * N + cols

    x = tl.load(ptr).to(tl.float32)

    # ---- RMSNorm (fp32 accumulate, round to bf16, then * weight, round to bf16) ----
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    xn = (x * inv_rms).to(tl.bfloat16).to(tl.float32)
    rw = tl.load(RW + cols).to(tl.float32)
    x = (xn * rw).to(tl.bfloat16).to(tl.float32)

    # ---- ReLU ----
    x = tl.maximum(x, 0.0)

    # ---- LayerNorm (fp32 compute, biased var, round to bf16) ----
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    xhat = d * tl.rsqrt(var + 1e-5)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    x = (xhat * g + b).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact, erf) twice, rounding to bf16 between like PyTorch ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    tl.store(Y + row * N + cols, x.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores on A100)
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _fused_rms_relu_ln_gelu2[(Mrows,)](
            x, self.rms1_w, self.ln3_g, self.ln3_b, y,
            N=N, BLOCK=N,
            num_warps=4,
        )
        return y
