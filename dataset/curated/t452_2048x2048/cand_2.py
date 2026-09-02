import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 452
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X_ptr, W2_ptr, W3_ptr, G_ptr, B_ptr, Y_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul output (bf16) -> fp32
    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.499  (bf16 op with fp32 opmath, rounded back to bf16)
    x = (x * 1.499).to(tl.bfloat16).to(tl.float32)

    # RMSNorm #1: fp32 mean-of-squares, cast normalized value to bf16,
    # then bf16*bf16 weight multiply (fp32 opmath, rounded to bf16)
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # RMSNorm #2
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    x = (x * w3).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation, biased var, eps=1e-5)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS/tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(Mrows,)](
            h, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b, y,
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
