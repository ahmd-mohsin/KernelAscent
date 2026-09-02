import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 207
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_bias_gelu_rms_kernel(
    X, B, W, Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    x = tl.load(X + base + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)

    # x + b1  (fp16 rounding as in reference)
    t = (x + b).to(tl.float16).to(tl.float32)

    # * 1.0905 (fp16 rounding)
    u = (t * 1.0905).to(tl.float16).to(tl.float32)

    # exact GELU computed in fp32, rounded to fp16 (matches PyTorch half gelu opmath)
    g = 0.5 * u * (1.0 + tl.math.erf(u * 0.7071067811865476))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(gf * gf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    n = (gf * r).to(tl.float16).to(tl.float32)

    w = tl.load(W + offs).to(tl.float32)
    o = (n * w).to(tl.float16)

    tl.store(Out + base + offs, o)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = torch.matmul(x, self.W0)

        M_, N_ = y.shape
        out = torch.empty_like(y)

        grid = (M_,)
        _fused_bias_gelu_rms_kernel[grid](
            y, self.b1, self.rms4_w, out,
            N=N_, BLOCK=N_,
            num_warps=8,
        )
        return out
