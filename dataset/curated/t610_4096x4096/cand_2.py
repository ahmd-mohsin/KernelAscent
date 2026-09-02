import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 610
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _ln_bias_gelu_relu_kernel(
    X, G, B, B2, Y,
    N, stride_xm, stride_ym,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch bf16 layer_norm internal fp32 math)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b

    # round to bf16 (layer_norm output), then add b2 in bf16
    y_bf = y.to(tl.bfloat16)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    z_bf = y_bf + b2

    # gelu (exact erf) computed in fp32, rounded to bf16
    z = z_bf.to(tl.float32)
    gel = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    gel_bf = gel.to(tl.bfloat16)

    # relu
    zero = tl.zeros_like(gel_bf)
    out = tl.maximum(gel_bf, zero)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _ln_bias_gelu_relu_kernel[(m,)](
            x, self.ln1_g, self.ln1_b, self.b2, y,
            n, x.stride(0), y.stride(0),
            1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
