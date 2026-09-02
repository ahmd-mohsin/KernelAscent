import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 846
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_softmax_kernel(
    X_ptr, B1_ptr, B2_ptr, Y_ptr,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Replicate PyTorch half-precision elementwise semantics:
    # each op computes in fp32 then rounds to fp16.
    t = (x * 1.1929).to(tl.float16).to(tl.float32)
    t = (t + b1).to(tl.float16).to(tl.float32)
    t = (t + b2).to(tl.float16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch's internal accumulation for half)
    t = tl.where(mask, t, float("-inf"))
    row_max = tl.max(t, axis=0)
    e = tl.exp(t - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            t = x * 1.1929
            t = t + self.b1
            t = t + self.b2
            return torch.softmax(t, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_softmax_kernel[(m,)](
            x, self.b1, self.b2, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
