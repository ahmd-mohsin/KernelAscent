import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 207
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_bias_gelu_rms(X, B, W, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    x = tl.load(ptr).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)

    # x = x + b1  (half op with float opmath, rounded to fp16)
    x = (x + b).to(tl.float16).to(tl.float32)
    # x = x * 1.0905
    x = (x * 1.0905).to(tl.float16).to(tl.float32)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(g * g, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    r = (g * inv).to(tl.float16).to(tl.float32)

    w = tl.load(W + offs).to(tl.float32)
    y = (r * w).to(tl.float16)
    tl.store(Y + row * N + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        _fused_bias_gelu_rms[(m,)](
            h, self.b1, self.rms4_w, y, n,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
