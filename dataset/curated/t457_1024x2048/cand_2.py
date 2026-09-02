import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 457
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_scale_rms_gelu(X, W, Y, N, eps, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load x (bf16) -> f32
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.2772 (computed in f32, rounded back to bf16 like PyTorch)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in float32
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    y = (x * r).to(tl.bfloat16).to(tl.float32)

    # multiply by weight (bf16 mul semantics: f32 opmath, round to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    t = (y * w).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), computed in f32 then rounded to bf16 (matches PyTorch)
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))

    tl.store(Y + row * N + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 via cuBLAS
        h = x @ self.W0
        h = h.contiguous()
        Mr, N = h.shape

        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_rms_gelu[(Mr,)](
            h, self.rms2_w, out,
            N, 1e-6, 1.2772,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2 via cuBLAS
        return out @ self.W4
