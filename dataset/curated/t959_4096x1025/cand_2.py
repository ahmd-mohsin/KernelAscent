import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 959
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_ln_ln_bias_relu_softmax(
    X_ptr, G1_ptr, B1_ptr, G2_ptr, B2_ptr, B3_ptr, Y_ptr,
    N: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(X_ptr + row * N + offs).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, bf16 rounding of output, like PyTorch) ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = x - mean1
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(G1_ptr + offs).to(tl.float32)
    b1 = tl.load(B1_ptr + offs).to(tl.float32)
    y = d1 * rstd1 * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(y, axis=0) / N
    d2 = y - mean2
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2_ptr + offs).to(tl.float32)
    b2 = tl.load(B2_ptr + offs).to(tl.float32)
    z = d2 * rstd2 * g2 + b2
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- + b3 (bf16 rounding), ReLU ----
    b3 = tl.load(B3_ptr + offs).to(tl.float32)
    z = (z + b3).to(tl.bfloat16).to(tl.float32)
    z = tl.maximum(z, 0.0)

    # ---- Softmax (fp32 math, bf16 output, like PyTorch) ----
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * N + offs, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows = h.shape[0]
        out = torch.empty_like(h)
        _fused_ln_ln_bias_relu_softmax[(rows,)](
            h, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, out,
            N=512, EPS=1e-5,
            num_warps=4,
        )
        return out
