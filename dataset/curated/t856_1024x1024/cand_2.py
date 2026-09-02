import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 856
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _gelu_ln_bias_kernel(
    X, G, B, B3, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

    # layernorm stats in fp32
    g_masked = tl.where(mask, g, 0.0)
    mean = tl.sum(g_masked, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)

    y = (g - mean) * rstd * gamma + beta
    # match reference: layernorm output cast to fp16, then bias add in fp16
    y = y.to(tl.float16).to(tl.float32) + b3
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _gelu_ln_bias_kernel[(m,)](
            h, self.ln2_g, self.ln2_b, self.b3, y,
            n, h.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
