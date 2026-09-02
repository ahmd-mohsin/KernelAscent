import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 950
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _rms2_gelu_kernel(X, W1, W2, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load row in fp32 (input is bf16)
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # --- RMSNorm 1 (compute in fp32, round to bf16, then weighted mul rounded to bf16) ---
    ms = tl.sum(x * x, axis=0) / N
    rs = tl.math.rsqrt(ms + 1e-6)
    x = (x * rs).to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w1).to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm 2 ---
    ms = tl.sum(x * x, axis=0) / N
    rs = tl.math.rsqrt(ms + 1e-6)
    x = (x * rs).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # --- exact GELU (erf), fp32 opmath like PyTorch, cast back to bf16 ---
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(Y + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms2_gelu_kernel[(m,)](
            x, self.rms1_w, self.rms2_w, y,
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
