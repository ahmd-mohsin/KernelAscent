import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 529
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_bias_rms_kernel(
    X, B, W, Y,
    N: tl.constexpr,
    stride_xm, stride_ym,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp32 then round to bf16 (matches bf16 add semantics)
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    xn = (xf * inv).to(tl.bfloat16)  # round to bf16 like .to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)  # bf16 mul
    y = (y.to(tl.float32) * SCALE).to(tl.bfloat16)              # bf16 scale

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # tensor-core bf16 matmul
        m, n = x.shape
        y = torch.empty_like(x)
        _fused_bias_rms_kernel[(m,)](
            x, self.b1, self.rms2_w, y,
            n,
            x.stride(0), y.stride(0),
            1e-6, 1.181,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
