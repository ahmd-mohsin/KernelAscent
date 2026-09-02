import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 52
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_rms_relu_bias_rms(
    X_ptr, W1_ptr, B3_ptr, W4_ptr, Y_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row (bf16 -> fp32)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1 (fp32 stats, round to bf16 before weight mul)
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)

    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w1).to(tl.bfloat16).to(tl.float32)

    # ReLU (exact in bf16)
    x = tl.maximum(x, 0.0)

    # bias add (bf16 rounding)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b3).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ms2 = tl.sum(x * x, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    x = (x * r2).to(tl.bfloat16).to(tl.float32)

    w4 = tl.load(W4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * w4).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_relu_bias_rms[(Mrows,)](
            h, self.rms1_w, self.b3, self.rms4_w, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
