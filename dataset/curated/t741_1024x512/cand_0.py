import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 741
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _scale_ln_kernel(
    X, Y, G, B,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # replicate bf16 rounding of the reference's `x * 1.106` (done in bf16)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gelu_kernel(X, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(X + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape

        # Fused scale + LayerNorm (fp32 accumulation, matching PyTorch semantics)
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _scale_ln_kernel[(Mrows,)](
            x, y, self.ln1_g, self.ln1_b,
            N, 1e-5, 1.106,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Matmul via cuBLAS tensor cores (bf16)
        out = y @ self.W2

        # In-place fused GELU (exact erf variant, fp32 math)
        n = out.numel()
        GB = 1024
        _gelu_kernel[(triton.cdiv(n, GB),)](out, n, BLOCK=GB, num_warps=4)
        return out
