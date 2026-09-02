import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 779
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_gelu_ln_softmax(X, G, B, Y, N, eps,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf variant), round to fp16 to match reference intermediate
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h = diff * inv * gamma + beta
    h = h.to(tl.float16).to(tl.float32)

    # Softmax in fp32
    h = tl.where(mask, h, float('-inf'))
    m = tl.max(h, axis=0)
    e = tl.exp(h - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS half-precision GEMM (tensor cores)
        x = x @ self.W0
        x = x.contiguous()

        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_gelu_ln_softmax[(m,)](
            x, self.ln2_g, self.ln2_b, y,
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
