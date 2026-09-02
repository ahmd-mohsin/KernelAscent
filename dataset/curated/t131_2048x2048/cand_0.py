import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 131
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _bias_softmax_ln_kernel(X, B, G, BETA, Y, N: tl.constexpr, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    # bias add in fp16 (matches x + b1 in half precision)
    x = tl.load(ptr)
    b = tl.load(B + offs)
    x = (x + b).to(tl.float32)

    # softmax in fp32 (matches torch's internal fp32 accumulation)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    p = e / s
    # round to fp16 (softmax output dtype in reference) before layernorm
    p = p.to(tl.float16).to(tl.float32)

    # layernorm in fp32
    mean = tl.sum(p, 0) / N
    d = p - mean
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs).to(tl.float32)
    be = tl.load(BETA + offs).to(tl.float32)
    y = d * rstd * g + be
    tl.store(Y + row * N + offs, y.to(tl.float16))


@triton.jit
def _ln_kernel(X, G, BETA, Y, N: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + offs).to(tl.float32)

    mean = tl.sum(x, 0) / N
    d = x - mean
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs).to(tl.float32)
    be = tl.load(BETA + offs).to(tl.float32)
    y = d * rstd * g + be
    tl.store(Y + row * N + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows = x.shape[0]
        N = self.W0.shape[1]  # 4096
        eps = 1e-5

        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0

        # fused: bias add + softmax + layernorm
        t = torch.empty_like(h)
        _bias_softmax_ln_kernel[(rows,)](
            h, self.b1, self.ln3_g, self.ln3_b, t,
            N, eps, BLOCK=N, num_warps=8,
        )

        # GEMM 2
        y = t @ self.W4

        # fused layernorm
        out = torch.empty_like(y)
        _ln_kernel[(rows,)](
            y, self.ln5_g, self.ln5_b, out,
            N, eps, BLOCK=N, num_warps=8,
        )
        return out
