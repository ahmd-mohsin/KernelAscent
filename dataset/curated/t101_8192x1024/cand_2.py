import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 101
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _relu_rms_w_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    # One program per row: fused relu -> rmsnorm(fp32) -> cast bf16 -> * weight (opmath fp32)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)  # relu (bf16 values unchanged by max with 0)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    y = (x * rstd).to(tl.bfloat16)  # match .to(x.dtype) rounding point
    w = tl.load(W + offs, mask=mask, other=0.0)
    out = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)  # bf16*bf16 opmath=fp32
    tl.store(Y + row * N + offs, out, mask=mask)


@triton.jit
def _relu_scale_kernel(X, Y, n_elements, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    x = tl.maximum(x, 0.0) * scale
    tl.store(Y + offs, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 via cuBLAS (bf16, TF32/TC on A100)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _relu_rms_w_kernel[(rows,)](
            h, self.rms2_w, y, N, 1e-6, BLOCK=BLOCK, num_warps=8
        )

        # GEMM 2 via cuBLAS
        z = y @ self.W3
        z = z.contiguous()
        out = torch.empty_like(z)
        n = z.numel()
        BLOCK2 = 1024
        _relu_scale_kernel[(triton.cdiv(n, BLOCK2),)](
            z, out, n, 1.0805, BLOCK=BLOCK2, num_warps=4
        )
        return out
