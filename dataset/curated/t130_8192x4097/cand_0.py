import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 130
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_ln_gelu_kernel(
    X, G, B, BIAS, Y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    # scale (bf16 elementwise mul semantics: f32 compute, round to bf16)
    x = tl.load(ptr).to(tl.float32)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    # softmax in f32, round output to bf16 (matches ATen bf16 softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # layer norm in f32 over the bf16-rounded softmax output
    mean = tl.sum(p, axis=0) / N
    d = p - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    y = (d * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU in f32, round to bf16
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.bfloat16).to(tl.float32)

    # bias add (f32 compute, bf16 round)
    bias = tl.load(BIAS + offs).to(tl.float32)
    out = (y + bias).to(tl.bfloat16)
    tl.store(Y + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # heavy matmul via cuBLAS tensor cores
        h = x @ self.W0
        h2 = h.contiguous().view(-1, h.shape[-1])
        rows, N = h2.shape
        out = torch.empty_like(h2)
        _fused_softmax_ln_gelu_kernel[(rows,)](
            h2, self.ln3_g, self.ln3_b, self.b5, out,
            N, 1e-5, 1.4882,
            BLOCK=N,
            num_warps=8,
        )
        return out.view(h.shape)
