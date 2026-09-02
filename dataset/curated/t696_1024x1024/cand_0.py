import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 696
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _relu_softmax_scale_kernel(
    X, Y,
    n_cols,
    stride_xm, stride_ym,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # relu in input dtype semantics (bf16 relu is exact), then upcast to fp32
    x = tl.maximum(x, 0.0).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.math.exp(x)
    num = tl.where(mask, num, 0.0)
    den = tl.sum(num, axis=0)
    sm = num / den

    # match: softmax output rounded to bf16, then scaled in fp32, rounded to bf16
    sm_bf16 = sm.to(tl.bfloat16)
    out = (sm_bf16.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        n_rows, n_cols = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 1024 else 4
        _relu_softmax_scale_kernel[(n_rows,)](
            h, y,
            n_cols,
            h.stride(0), y.stride(0),
            1.0435,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
