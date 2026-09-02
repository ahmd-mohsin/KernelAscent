import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 99
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_gelu_ln_softmax(X, G, B, Out, N, eps,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X + row * N + cols

    x = tl.load(ptr).to(tl.float32)

    # exact (erf-based) GELU
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # emulate fp16 intermediate rounding of the reference
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm (float32 accumulation, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # Softmax (float32 accumulation, like PyTorch)
    mx = tl.max(y, axis=0)
    e = tl.exp(y - mx)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out + row * N + cols, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_gelu_ln_softmax[(Mrows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, 1e-5,
            BLOCK=512,
            num_warps=4,
        )
        return out
