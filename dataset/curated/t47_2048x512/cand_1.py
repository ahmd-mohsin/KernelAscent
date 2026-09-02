import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 47
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_bias_ln_scale_rms_kernel(
    Y_ptr, B1_ptr, G_ptr, B_ptr, RW_ptr, OUT_ptr,
    N: tl.constexpr,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)

    # load matmul output row and bias; add in fp32, round to bf16 (matches x + b1 in bf16)
    y = tl.load(Y_ptr + row * N + offs).to(tl.float32)
    b1 = tl.load(B1_ptr + offs).to(tl.float32)
    t = (y + b1).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch bf16 layer_norm which computes in fp32)
    mean = tl.sum(t, axis=0) / N
    d = t - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.rsqrt(var + LN_EPS)

    g = tl.load(G_ptr + offs).to(tl.float32)
    beta = tl.load(B_ptr + offs).to(tl.float32)
    h = (d * rstd * g + beta).to(tl.bfloat16).to(tl.float32)

    # scale by 1.0003 (fp32 opmath, round to bf16 as PyTorch does)
    h = (h * 1.0003).to(tl.bfloat16).to(tl.float32)

    # RMSNorm: fp32 mean of squares, rsqrt, multiply, cast to bf16, then * weight
    ms = tl.sum(h * h, axis=0) / N
    r = tl.rsqrt(ms + RMS_EPS)
    w = tl.load(RW_ptr + offs).to(tl.float32)
    out = ((h * r).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16)

    tl.store(OUT_ptr + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        y = x @ self.W0  # (M, 4096) bf16
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)

        _fused_bias_ln_scale_rms_kernel[(Mrows,)](
            y, self.b1, self.ln2_g, self.ln2_b, self.rms4_w, out,
            N=N,
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            num_warps=8,
        )
        return out
