import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 741
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _scale_ln_kernel(X, G, B, Y, N, eps, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * scale
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = x * 1.106
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x @ self.W2
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]

        # Fused: scale-by-1.106 + LayerNorm (fp32 accumulation, bf16 output)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _scale_ln_kernel[(rows,)](
            x2, self.ln1_g, self.ln1_b, y,
            N, 1e-5, 1.106,
            BLOCK=BLOCK, num_warps=8,
        )

        # GEMM via cuBLAS (tensor cores)
        out = y @ self.W2

        # Fused exact-erf GELU
        z = torch.empty_like(out)
        n = out.numel()
        GB = 1024
        _gelu_kernel[(triton.cdiv(n, GB),)](out, z, n, BLOCK=GB, num_warps=4)

        return z.view(*orig_shape[:-1], self.W2.shape[1])
