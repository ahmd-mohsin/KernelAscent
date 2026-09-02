import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    S1: tl.constexpr,   # 1.0809
    S2: tl.constexpr,   # 1.4484
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (fp16 -> fp32)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # scale by 1.0809 (fp16 rounding as in eager op), then relu
    x = (x * S1).to(tl.float16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # layer norm (fp32 statistics, matches ATen half layer_norm accumulation)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # scale by 1.4484 (fp16 rounding)
    y = (y * S2).to(tl.float16).to(tl.float32)

    # RMS norm in fp32, cast to fp16, then multiply by weight (fp16)
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    yn = (y * r).to(tl.float16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (yn * w).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            y, self.ln3_g, self.ln3_b, self.rms5_w, out,
            N=N,
            S1=1.0809, S2=1.4484,
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
