import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 733
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_post_kernel(
    X, W2, G3, B3, W4, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # scale by 1.4626 (computed in fp32, rounded to fp16 like PyTorch half op)
    x = (x * 1.4626).to(tl.float16).to(tl.float32)

    # RMSNorm #1 (fp32 math, cast to fp16, then fp16*fp16 weight mul in fp32 opmath)
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 internal math, eps=1e-5)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xc * rstd * g3 + b3).to(tl.float16).to(tl.float32)

    # RMSNorm #2
    ms2 = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms2 + 1e-6)).to(tl.float16).to(tl.float32)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w4).to(tl.float16).to(tl.float32)

    # GELU (erf variant, fp32 opmath)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(Y + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        x = torch.matmul(x, self.W0)
        x = x.contiguous()

        M_, N_ = x.shape
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(N_)
        _fused_post_kernel[(M_,)](
            x, self.rms2_w, self.ln3_g, self.ln3_b, self.rms4_w, y,
            N=N_, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
