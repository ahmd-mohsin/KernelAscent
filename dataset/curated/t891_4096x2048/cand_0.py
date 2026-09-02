import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 891
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _softmax_ln_kernel(
    X, Y, G, B,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    x = tl.load(ptr).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch fp16 softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to fp16 (softmax output dtype), then layernorm reads fp16 values
    p = p.to(tl.float16).to(tl.float32)

    # layernorm (fp32 accumulation)
    mean = tl.sum(p, axis=0) / N
    d = p - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y + row * N + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        _softmax_ln_kernel[(rows,)](
            h, y, self.ln2_g, self.ln2_b,
            N, 1e-5,
            BLOCK=N,
            num_warps=8,
            num_stages=1,
        )
        return y
