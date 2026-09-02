import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 689
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    eps,
    scale,
    BLOCK: tl.constexpr,
    APPLY_RELU_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # mean of squares in fp32 (matches _xf.pow(2).mean(-1))
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    # normalize in fp32, round to fp16 (matches .to(x.dtype))
    xn = (x * inv).to(tl.float16)

    # weight multiply: fp16*fp16 done in fp32 opmath, rounded back (matches CUDA half mul)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn.to(tl.float32) * w).to(tl.float16)

    if APPLY_RELU_SCALE:
        # relu (exact in fp16)
        y = tl.where(y > 0, y, y * 0)
        # scalar mul in fp32 opmath, round back to fp16 (matches PyTorch half*scalar)
        y = (y.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y_ptr + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape

        # Fused: RMSNorm(x)*rms0_w -> relu -> *1.0454
        y1 = torch.empty_like(x)
        BLOCK1 = triton.next_power_of_2(n)
        _rmsnorm_kernel[(m,)](
            x, self.rms0_w, y1,
            n, 1e-6, 1.0454,
            BLOCK=BLOCK1,
            APPLY_RELU_SCALE=True,
            num_warps=8,
        )

        # GEMM on tensor cores
        y2 = y1 @ self.W3
        y2 = y2.contiguous()
        n2 = y2.shape[1]

        # Fused: RMSNorm(y2)*rms4_w
        out = torch.empty_like(y2)
        BLOCK2 = triton.next_power_of_2(n2)
        _rmsnorm_kernel[(m,)](
            y2, self.rms4_w, out,
            n2, 1e-6, 1.0,
            BLOCK=BLOCK2,
            APPLY_RELU_SCALE=False,
            num_warps=4,
        )
        return out
