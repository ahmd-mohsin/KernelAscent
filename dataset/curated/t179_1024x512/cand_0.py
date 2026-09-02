import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 179
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, RW, G3, B3, G4, B4, Y,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (fp32 math, cast to fp16, then affine in fp16) ----
    ms = tl.sum(x * x, axis=0) / D
    y = (x * (1.0 / tl.sqrt(ms + 1e-6))).to(tl.float16)
    rw = tl.load(RW + cols, mask=mask, other=0.0)
    y = y * rw  # fp16 multiply (matches reference)
    # scalar multiply: PyTorch does fp16 * python-float in fp32 opmath then casts
    y = (y.to(tl.float32) * 1.4245).to(tl.float16)

    # ---- LayerNorm 3 (fp32 accumulation, fp16 output) ----
    xf = y.to(tl.float32)
    mean = tl.sum(xf, axis=0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xf - mean) * rstd * g3 + b3).to(tl.float16)

    # ---- LayerNorm 4 (fp32 accumulation, fp16 output) ----
    xf = y.to(tl.float32)
    mean = tl.sum(xf, axis=0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((xf - mean) * rstd * g4 + b4).to(tl.float16)

    tl.store(Y + row * D + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores (already optimal)
        x = x @ self.W0
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_norms_kernel[(m,)](
            x, self.rms1_w, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, y,
            D=d, BLOCK=BLOCK, num_warps=4,
        )
        return y
