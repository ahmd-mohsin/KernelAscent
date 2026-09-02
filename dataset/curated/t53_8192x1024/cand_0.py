import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 53
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _rmsnorm_scale_kernel(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # normalize, round to bf16 (matches .to(x.dtype))
    xn = (x * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)

    # each PyTorch elementwise op computes in fp32 (opmath) then rounds to bf16
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.4888).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.1305).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.0234).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_scale_kernel[(Mrows,)](
            x, self.rms0_w, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
