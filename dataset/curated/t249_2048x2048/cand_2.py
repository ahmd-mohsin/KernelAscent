import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 249
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS norm (computed in fp32, like the reference)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    # match reference dtype semantics: round to bf16 after each bf16 op
    y = (x * rstd).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    y = (y * scale).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 opmath like PyTorch's bf16 kernel
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores on A100)
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _rms_gelu_kernel[(m,)](
            x, self.rms1_w, y,
            n, x.stride(0), y.stride(0),
            1e-6, 1.2502,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
