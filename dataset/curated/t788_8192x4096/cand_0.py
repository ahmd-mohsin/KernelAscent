import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 788
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _ln3_gelu_kernel(
    X, Y,
    G1, B1, G2, B2, G3, B3,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X + row * N + cols

    x = tl.load(ptr).to(tl.float32)

    g1 = tl.load(G1 + cols).to(tl.float32)
    b1 = tl.load(B1 + cols).to(tl.float32)
    g2 = tl.load(G2 + cols).to(tl.float32)
    b2 = tl.load(B2 + cols).to(tl.float32)
    g3 = tl.load(G3 + cols).to(tl.float32)
    b3 = tl.load(B3 + cols).to(tl.float32)

    inv_n = 1.0 / N

    # LayerNorm 1
    m = tl.sum(x, axis=0) * inv_n
    d = x - m
    v = tl.sum(d * d, axis=0) * inv_n
    x = d * (1.0 / tl.sqrt(v + eps)) * g1 + b1
    x = x.to(tl.float16).to(tl.float32)  # match intermediate fp16 rounding

    # LayerNorm 2
    m = tl.sum(x, axis=0) * inv_n
    d = x - m
    v = tl.sum(d * d, axis=0) * inv_n
    x = d * (1.0 / tl.sqrt(v + eps)) * g2 + b2
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm 3
    m = tl.sum(x, axis=0) * inv_n
    d = x - m
    v = tl.sum(d * d, axis=0) * inv_n
    x = d * (1.0 / tl.sqrt(v + eps)) * g3 + b3
    x = x.to(tl.float16).to(tl.float32)

    # GELU (exact, erf-based)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(Y + row * N + cols, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS (tensor cores) for the matmul
        h = torch.matmul(x, self.W0)

        rows, N = h.shape
        y = torch.empty_like(h)

        _ln3_gelu_kernel[(rows,)](
            h, y,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            N, 1e-5,
            BLOCK=1024,
            num_warps=8,
        )
        return y
