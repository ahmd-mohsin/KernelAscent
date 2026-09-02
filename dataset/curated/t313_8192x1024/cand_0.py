import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 313
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _relu_ln_kernel(X, G, B, Y, D: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * D + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gelu_scale_kernel(X, N, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU, computed in fp32 like PyTorch's bf16 opmath
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476)) * scale
    tl.store(X + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]

        # Fused ReLU + LayerNorm (one program per row)
        h = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _relu_ln_kernel[(m,)](
            x2, self.ln1_g, self.ln1_b, h,
            D=d, eps=1e-5, BLOCK=BLOCK,
            num_warps=8,
        )

        # cuBLAS matmul (tensor cores)
        out = h @ self.W2

        # Fused GELU + scale, in-place on matmul output
        n = out.numel()
        EBLOCK = 1024
        grid = (triton.cdiv(n, EBLOCK),)
        _gelu_scale_kernel[grid](out, n, 1.1733, BLOCK=EBLOCK, num_warps=4)

        return out.view(*orig_shape[:-1], self.W2.shape[1])
