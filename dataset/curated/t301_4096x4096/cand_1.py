import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 301
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_softmax_ln_relu(X, G, B, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    x = tl.load(ptr).to(tl.float32)
    # scale (match fp16 rounding of reference: fp16 * scalar -> fp16)
    x = (x * 1.0477).to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, as PyTorch does internally)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    p = e / s
    # softmax output is stored as fp16 in reference before layer_norm reads it
    p = p.to(tl.float16).to(tl.float32)

    # layer norm (fp32 accumulation)
    mean = tl.sum(p, 0) / N
    d = p - mean
    var = tl.sum(d * d, 0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    y = d * inv * g + b

    # relu
    y = tl.maximum(y, 0.0)

    tl.store(Out + row * N + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_softmax_ln_relu[(Mrows,)](
            h, self.ln3_g, self.ln3_b, out,
            N=N, BLOCK=N,
            num_warps=8,
        )
        return out
