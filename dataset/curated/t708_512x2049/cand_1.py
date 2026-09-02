import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 708
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _gelu_rms_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU, computed in fp32 then rounded to bf16 (matches F.gelu on bf16)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    # RMS norm in fp32
    ms = tl.sum(g * g, axis=0) / N
    rstd = tl.math.rsqrt(ms + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g * rstd).to(tl.bfloat16).to(tl.float32) * w
    tl.store(Y + row * N + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _scale_gelu_kernel(X, Y, n_elements, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # scale in fp32, round to bf16 (matches bf16 tensor * python scalar)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS bf16 tensor cores)
        h = x @ self.W0  # (M, 2048), bf16, contiguous

        rows, N = h.shape
        h = h.contiguous()
        y = torch.empty_like(h)

        # Fused: exact GELU -> RMSNorm -> weight scale
        BLOCK = triton.next_power_of_2(N)
        _gelu_rms_kernel[(rows,)](
            h, self.rms2_w, y, N, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )

        # GEMM 2
        z = y @ self.W3  # (M, 2048), bf16

        # Fused: scale -> exact GELU
        z = z.contiguous()
        out = torch.empty_like(z)
        n = z.numel()
        BLOCK2 = 1024
        grid = (triton.cdiv(n, BLOCK2),)
        _scale_gelu_kernel[grid](z, out, n, 1.3239, BLOCK=BLOCK2, num_warps=4)

        return out
