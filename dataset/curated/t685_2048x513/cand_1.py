import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 685
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _softmax_scale_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # match reference: softmax output rounded to bf16, then scalar mul
    # (CUDA elementwise mul upcasts bf16 to fp32 internally), then rounded back
    sm_bf16 = sm.to(tl.bfloat16)
    out = (sm_bf16.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_scale_kernel[(m,)](
            h, y,
            h.stride(0), y.stride(0),
            n,
            SCALE=1.0138,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
