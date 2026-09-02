import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 64
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X_ptr, W1_ptr, W2_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # First RMSNorm (fp32 accumulation, round to bf16, then bf16 weight mul)
    ms1 = tl.sum(xf * xf, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + eps)
    x1 = (xf * r1).to(tl.bfloat16)
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0)
    x1 = x1 * w1  # bf16 multiply (fp32 mul + round-to-nearest bf16, same as torch)

    # Second RMSNorm
    x1f = x1.to(tl.float32)
    ms2 = tl.sum(x1f * x1f, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + eps)
    x2 = (x1f * r2).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0)
    y = x2 * w2

    tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        x = x @ self.W0

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]

        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _double_rmsnorm_kernel[(rows,)](
            x2d, self.rms1_w, self.rms2_w, y,
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
