import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 875
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _ln_gelu_bias_scale_kernel(
    X, G, B, B3, Y,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    # round to bf16 (layer_norm output dtype)
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf) in fp32, round back to bf16
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.bfloat16).to(tl.float32)

    # add bias, round to bf16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)

    # scale
    y = y * scale

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _ln_gelu_bias_scale_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.b3, out,
            N, h.stride(0), out.stride(0),
            1e-5, 1.3811,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
