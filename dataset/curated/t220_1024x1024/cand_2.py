import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 220
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, B3, G, B, B5, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # bf16

    # relu (bf16)
    x = tl.maximum(x, 0.0)

    # exact gelu in fp32, round back to bf16 (matches PyTorch opmath behavior)
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # + b3 in bf16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    h = (g + b3).to(tl.bfloat16)

    # layer norm: stats and affine in fp32
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    mean = tl.sum(hf, axis=0) / N
    diff = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (hf - mean) * inv_std * gamma + beta
    ln = ln.to(tl.bfloat16)

    # + b5 in bf16
    b5 = tl.load(B5 + cols, mask=mask, other=0.0)
    out = (ln + b5).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core matmul (cuBLAS)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            h, self.b3, self.ln4_g, self.ln4_b, self.b5, y,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return y
