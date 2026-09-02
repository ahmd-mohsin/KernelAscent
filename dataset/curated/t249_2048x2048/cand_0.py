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
    X_ptr, W_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    r = x * inv

    # round to bf16 (matches .to(x.dtype))
    r = r.to(tl.bfloat16).to(tl.float32)

    # multiply by weight (bf16 op with fp32 opmath, rounded to bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    a = (r * w).to(tl.bfloat16).to(tl.float32)

    # scalar multiply (fp32 opmath, rounded to bf16)
    b = (a * scale).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based) in fp32, rounded to bf16
    g = b * 0.5 * (1.0 + tl.math.erf(b * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_ym + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mr, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_gelu_kernel[(Mr,)](
            x, self.rms1_w, y,
            N, x.stride(0), y.stride(0),
            1e-6, 1.2502,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
