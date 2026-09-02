import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 353
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_ln_softmax_kernel(
    X_ptr, G_ptr, B_ptr, B5_ptr, Y_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- exact GELU (erf), computed in fp32, rounded to fp16 like PyTorch ----
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # ---- LayerNorm (stats in fp32, output rounded to fp16 like PyTorch) ----
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * gamma + beta
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, output rounded to fp16) ----
    y_masked = tl.where(mask, y, float("-inf"))
    m = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - m)
    e = tl.where(mask, e, 0.0)
    s = e / tl.sum(e, axis=0)
    s = s.to(tl.float16).to(tl.float32)

    # ---- scale (fp32 opmath, round to fp16) ----
    s = (s * 1.2464).to(tl.float16).to(tl.float32)

    # ---- bias add (fp32 opmath, round to fp16) ----
    b5 = tl.load(B5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (s + b5).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_softmax_kernel[(rows,)](
            h, self.ln2_g, self.ln2_b, self.b5, y,
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
