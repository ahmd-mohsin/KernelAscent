import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 96
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _ln_scale_relu_kernel(
    X, G, B, Y,
    stride_xm, stride_ym,
    N, eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # round to bf16 (matches layer_norm output dtype), then scale in fp32, round again
    y = y.to(tl.bfloat16).to(tl.float32)
    y = y * scale
    y = y.to(tl.bfloat16)
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _ln_scale_relu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, out,
            h.stride(0), out.stride(0),
            N, 1e-5, 1.2036,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
