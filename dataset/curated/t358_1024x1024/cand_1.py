import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 358
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _rmsnorm_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (x * rstd).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fused matmul + bias (single cuBLAS epilogue), then second matmul
        h = torch.addmm(self.b1, x, self.W0)
        z = h @ self.W2

        z = z.contiguous()
        Mrows, N = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_kernel[(Mrows,)](
            z, self.rms3_w, out,
            z.stride(0), out.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
