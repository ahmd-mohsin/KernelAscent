import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 940
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_bias_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # bias add in bf16 (matches x + b0 + b1 done in bf16), bias pre-summed in bf16
    x = (x + b).to(tl.bfloat16)

    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_ym + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self._b_sum = None

    def forward(self, x):
        if self._b_sum is None or self._b_sum.device != x.device:
            # (b0 + b1) computed in bf16, same as sequential bf16 adds order:
            # (x + b0) + b1 == x + (b0 + b1) only approximately; to be exact we must
            # apply them sequentially. But bf16 add of two constants then add to x
            # differs. So keep separate adds fused in kernel instead.
            self._b_sum = None
        if not x.is_cuda:
            x = x + self.b0
            x = x + self.b1
            return torch.softmax(x, dim=-1)

        Mrows, N = x.shape
        x = x.contiguous()
        y = torch.empty_like(x)

        # Precompute per-element (x + b0) + b1 exactly by fusing two bf16 adds
        # inside the kernel: we pass b0 and b1 separately.
        BLOCK = triton.next_power_of_2(N)
        grid = (Mrows,)
        _fused_bias_softmax_2_kernel[grid](
            x, self.b0, self.b1, y,
            N, x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y


@triton.jit
def _fused_bias_softmax_2_kernel(
    X_ptr, B0_ptr, B1_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b0 = tl.load(B0_ptr + cols, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0)

    # sequential bf16 rounding to match (x + b0) + b1 in bf16
    x = (x + b0).to(tl.bfloat16)
    x = (x + b1).to(tl.bfloat16)

    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_ym + cols, y.to(tl.bfloat16), mask=mask)
