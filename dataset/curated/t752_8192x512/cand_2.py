import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 752
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _relu_ln_bias_scale_kernel(
    X, G, B, B4, Y,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # relu (applied twice == once)
    x = tl.maximum(x, 0.0)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # round to bf16 (matches F.layer_norm output dtype)
    y = y.to(tl.bfloat16)

    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = (y.to(tl.float32) + b4.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core matmul
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_ln_bias_scale_kernel[(m,)](
            h, self.ln3_g, self.ln3_b, self.b4, y,
            n, h.stride(0), y.stride(0),
            1e-5, 1.4319,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
