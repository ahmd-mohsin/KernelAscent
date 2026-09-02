import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 567
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_norm_act_kernel(
    X_ptr, RMSW_ptr, G_ptr, B_ptr, Out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32, cast to bf16 (matches (_xf * rsqrt(...)).to(bf16))
    ms = tl.sum(x * x, axis=0) / N
    xn = x * tl.math.rsqrt(ms + 1e-6)
    xn = xn.to(tl.bfloat16).to(tl.float32)

    # multiply by rms weight (opmath fp32, rounded to bf16)
    w = tl.load(RMSW_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)

    # relu (exact on bf16 values)
    y = tl.maximum(y, 0.0)

    # LayerNorm: stats in fp32, affine in fp32, output rounded to bf16
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = d * inv * g + b
    z = z.to(tl.bfloat16).to(tl.float32)

    # GELU (erf) twice, computed in fp32, rounded to bf16 between
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    z = z * 0.5 * (1.0 + tl.math.erf(z * INV_SQRT2))
    z = z.to(tl.bfloat16).to(tl.float32)
    z = z * 0.5 * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(Out_ptr + row * N + offs, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.view(-1, N)
        rows = h2.shape[0]

        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_norm_act_kernel[(rows,)](
            h2, self.rms1_w, self.ln3_g, self.ln3_b, out,
            N=N, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
