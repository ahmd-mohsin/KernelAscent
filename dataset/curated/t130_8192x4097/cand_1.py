import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 130
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_ln_gelu_bias(
    X, G, B, B5, Out,
    stride_x,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X + row * stride_x + cols

    # load matmul result, apply scale (rounded to bf16 like the eager op does)
    x = tl.load(ptr).to(tl.float32)
    y = (x * 1.4882).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, output rounded to bf16 (matches eager intermediate)
    m = tl.max(y, axis=0)
    e = tl.math.exp(y - m)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # layer norm in fp32
    mean = tl.sum(p, axis=0) / N
    d = p - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    z = (d * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) gelu
    gel = (0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)

    # bias add
    b5 = tl.load(B5 + cols).to(tl.float32)
    out = (gel + b5).to(tl.bfloat16)
    tl.store(Out + row * stride_x + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference path
            h = x @ self.W0
            h = h * 1.4882
            h = torch.softmax(h, dim=-1)
            h = F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)
            h = F.gelu(h)
            return h + self.b5

        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)
        _fused_softmax_ln_gelu_bias[(rows,)](
            h, self.ln3_g, self.ln3_b, self.b5, out,
            h.stride(0),
            N=N,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
