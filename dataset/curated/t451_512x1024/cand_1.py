import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 451
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_bias_relu_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add + relu in bf16 (matches reference elementwise ops)
    v = x + b
    zero = tl.zeros_like(v)
    v = tl.maximum(v, zero)

    # softmax with fp32 accumulation (matches torch internal upcast)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N_)
        _fused_bias_relu_softmax[(M_,)](
            x, self.b0, y,
            x.stride(0), y.stride(0),
            N_, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
