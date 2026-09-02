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
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # (xf * rsqrt).to(bf16)
    y = (x * inv).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # bf16 mul (fp32 accumulate, round to bf16)
    z = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    # bf16 add
    z = (z.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # exact gelu at fp32 opmath, output bf16
    zf = z.to(tl.float32)
    g = 0.5 * zf * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_bias_gelu_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
