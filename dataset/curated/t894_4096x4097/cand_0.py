import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 894
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _rms_bias_gelu_kernel(
    X, W, B, Y,
    stride_xm,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + row * stride_xm + cols).to(tl.float32)

    # RMS norm (computed in fp32, like reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)  # cast to bf16 like reference

    w = tl.load(W + cols)
    b = tl.load(B + cols)

    # bf16 elementwise ops with fp32 opmath then round back (matches PyTorch)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # exact GELU (erf-based), computed in fp32
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    tl.store(Y + row * N + cols, g.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM with fp32 accumulate
        m, n = h.shape
        out = torch.empty((m, n), dtype=h.dtype, device=h.device)
        _rms_bias_gelu_kernel[(m,)](
            h, self.rms1_w, self.b2, out,
            h.stride(0),
            N=n,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
