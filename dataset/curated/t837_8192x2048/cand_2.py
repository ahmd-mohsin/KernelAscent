import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 837
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X_ptr, W1_ptr, W2_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # First RMSNorm (stats in fp32, normalized value cast to bf16, then * w1 in bf16)
    ms1 = tl.sum(x * x, axis=0) / N
    xn = (x * tl.math.rsqrt(ms1 + eps)).to(tl.bfloat16)
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0)
    x1 = (xn.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # Second RMSNorm
    xf = x1.to(tl.float32)
    ms2 = tl.sum(xf * xf, axis=0) / N
    xn2 = (xf * tl.math.rsqrt(ms2 + eps)).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0)
    y = (xn2.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (tensor cores via cuBLAS)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _double_rmsnorm_kernel[(Mrows,)](
            h, self.rms1_w, self.rms2_w, out,
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2
        return out @ self.W3
