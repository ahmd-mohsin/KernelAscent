import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 537
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_ln_ln_bias_gelu_relu(
    X, G1, B1, G2, B2, B3, Y,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    x = tl.load(ptr).to(tl.float32)

    # LayerNorm 1 (fp32 accumulation, like PyTorch on bf16)
    mean1 = tl.sum(x, axis=0) / N
    xc = x - mean1
    var1 = tl.sum(xc * xc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(G1 + offs).to(tl.float32)
    b1 = tl.load(B1 + offs).to(tl.float32)
    y = xc * rstd1 * g1 + b1
    # round to bf16 (intermediate tensor dtype in reference)
    y = y.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(y, axis=0) / N
    yc = y - mean2
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + offs).to(tl.float32)
    b2 = tl.load(B2 + offs).to(tl.float32)
    z = yc * rstd2 * g2 + b2
    z = z.to(tl.bfloat16).to(tl.float32)

    # + b3 (bf16 rounding as in reference)
    b3 = tl.load(B3 + offs).to(tl.float32)
    z = z + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), fp32 opmath then round to bf16
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ReLU
    g = tl.maximum(g, 0.0)

    tl.store(Y + row * N + offs, g.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 tensor-core matmul
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_ln_ln_bias_gelu_relu[(m,)](
            h, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, out,
            N=n, EPS=1e-5, BLOCK=2048,
            num_warps=8,
        )
        return out
