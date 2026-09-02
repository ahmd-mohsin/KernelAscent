import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 894
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(
    X, W, B, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # mean of squares over the row
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + eps)

    # replicate PyTorch's rounding steps (bf16 intermediates, fp32 opmath)
    normed = (xf * rrms).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    t = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    b = tl.load(B + cols, mask=mask, other=0.0)
    t2 = (t.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    tf = t2.to(tl.float32)
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * tf * (1.0 + tl.math.erf(tf * inv_sqrt2))

    tl.store(Y + row * stride_y + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_gelu_kernel[(m,)](
            x, self.rms1_w, self.b2, y,
            n, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
