import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 436
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _scale_rmsnorm_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N,
    eps,
    scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # x = x * 1.1783 (computed in fp32, rounded to fp16, matching PyTorch half kernel)
    xs = (x.to(tl.float32) * scale).to(tl.float16)
    xf = xs.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    xn = (xf * rstd).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # fp16 * fp16 with fp32 opmath, rounded back to fp16 (matches PyTorch)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _scale_rmsnorm_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            n,
            1e-6,
            1.1783,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y @ self.W2
