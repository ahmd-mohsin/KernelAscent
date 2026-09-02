import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 76
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_rms_gelu_ln_kernel(
    X_ptr, RW_ptr, G_ptr, B_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offs = row * N + cols

    # load row (fp16 -> fp32 for RMS stats, matching reference)
    x = tl.load(X_ptr + offs).to(tl.float32)

    # RMSNorm: stats in fp32, cast normalized value back to fp16, then * weight in fp16
    ms = tl.sum(x * x, axis=0) / N
    rinv = tl.math.rsqrt(ms + 1e-6)
    xh = (x * rinv).to(tl.float16)
    rw = tl.load(RW_ptr + cols)  # fp16
    xh = xh * rw                 # fp16 multiply (matches reference dtype behavior)

    # GELU (exact, erf-based) computed in fp32, cast back to fp16
    xf = xh.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    gh = g.to(tl.float16)

    # LayerNorm: stats and affine in fp32, output fp16
    gf = gh.to(tl.float32)
    mean = tl.sum(gf, axis=0) / N
    d = gf - mean
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    gamma = tl.load(G_ptr + cols).to(tl.float32)
    beta = tl.load(B_ptr + cols).to(tl.float32)
    y = (d * inv * gamma + beta) * 1.1664

    tl.store(Y_ptr + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        M_, N_ = x.shape
        y = torch.empty_like(x)
        _fused_rms_gelu_ln_kernel[(M_,)](
            x, self.rms1_w, self.ln3_g, self.ln3_b, y,
            N=N_, BLOCK=N_,
            num_warps=8,
        )
        return y
