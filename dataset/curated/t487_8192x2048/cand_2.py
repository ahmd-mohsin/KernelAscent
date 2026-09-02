import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 487
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_rms_rms_gelu(X, W1, W2, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * D + offs

    # first RMSNorm (stats in fp32)
    xf = tl.load(ptr).to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / D
    xn1 = xf * tl.math.rsqrt(ms1 + 1e-6)
    # cast to fp16, multiply by weight in fp16 (matches reference)
    x1 = xn1.to(tl.float16) * tl.load(W1 + offs)

    # second RMSNorm
    xf2 = x1.to(tl.float32)
    ms2 = tl.sum(xf2 * xf2, axis=0) / D
    xn2 = xf2 * tl.math.rsqrt(ms2 + 1e-6)
    x2 = xn2.to(tl.float16) * tl.load(W2 + offs)

    # exact GELU in fp32 (matches CUDA half gelu, which accumulates in float)
    g = x2.to(tl.float32)
    y = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))

    tl.store(Y + row * D + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        x = x @ self.W0
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        _fused_rms_rms_gelu[(m,)](
            x, self.rms1_w, self.rms2_w, y,
            D=d, BLOCK=d,
            num_warps=8,
        )
        return y
