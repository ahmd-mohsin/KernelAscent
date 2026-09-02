import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 233
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _ln_bias_gelu_kernel(
    X, G, B, B2, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm stats in fp32 (matches PyTorch bf16 layernorm accumulation)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    # round to bf16 as layer_norm output would be, then add bias (fp32 opmath)
    y = y.to(tl.bfloat16).to(tl.float32)

    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = y + b2
    z = z.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * z * (1 + erf(z / sqrt(2)))
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _ln_bias_gelu_kernel[(m,)](
            h, self.ln1_g, self.ln1_b, self.b2, y,
            h.stride(0), y.stride(0),
            n, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y
