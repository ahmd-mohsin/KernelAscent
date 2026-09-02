import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 957
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _rmsnorm_kernel(
    X, W, Y,
    N,
    eps,
    SCALE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0)  # fp16

    if APPLY_SCALE:
        # match PyTorch half-scalar mul: compute in fp32, round back to fp16
        x = (x.to(tl.float32) * SCALE).to(tl.float16)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    y16 = (xf * rstd).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)  # fp16
    out = y16 * w  # fp16 multiply (correctly rounded, same as fp32-compute-then-round)

    tl.store(Y + row * N + offs, out, mask=mask)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, scale: float, apply_scale: bool) -> torch.Tensor:
    x = x.contiguous()
    Mrows, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_kernel[(Mrows,)](
        x, w, y, N, 1e-6,
        SCALE=scale, APPLY_SCALE=apply_scale, BLOCK=BLOCK,
        num_warps=4,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)                      # cuBLAS fp16 tensor-core GEMM
        h = _rmsnorm(h, self.rms1_w, 1.0, False)          # fused RMSNorm
        h = torch.matmul(h, self.W2)                      # cuBLAS fp16 tensor-core GEMM
        h = _rmsnorm(h, self.rms4_w, 1.2112, True)        # fused scale + RMSNorm
        return h
