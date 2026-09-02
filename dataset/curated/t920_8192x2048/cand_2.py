import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 920
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_gelu_bias_scale_rmsnorm(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X_ptr + row * N + offs

    # load matmul output (fp16) and upcast
    x = tl.load(ptr).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to fp16 (matches PyTorch opmath)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # bias add: fp16 + fp16 (exact in fp32, then round -> identical to fp16 add)
    b = tl.load(B_ptr + offs).to(tl.float32)
    t16 = (g16.to(tl.float32) + b).to(tl.float16)

    # scalar multiply at float opmath then round to fp16 (matches PyTorch)
    s16 = (t16.to(tl.float32) * 1.3289).to(tl.float16)

    # RMSNorm in fp32
    xf = s16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    n16 = (xf * inv).to(tl.float16)

    # multiply by weight: fp16 * fp16 (exact in fp32, then round)
    w = tl.load(W_ptr + offs).to(tl.float32)
    out = (n16.to(tl.float32) * w).to(tl.float16)

    tl.store(Out_ptr + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already optimal on A100 tensor cores)
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        _fused_gelu_bias_scale_rmsnorm[(m,)](
            y, self.b2, self.rms4_w, out,
            N=n, BLOCK=n,
            num_warps=8,
        )
        return out
