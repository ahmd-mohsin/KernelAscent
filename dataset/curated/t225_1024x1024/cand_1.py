import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 225
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _softmax_bias_relu_kernel(
    X_ptr, B_ptr, Out_ptr,
    n_cols,
    stride_xm,
    stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # round softmax result to fp16 (matches torch.softmax output dtype)
    sm_h = sm.to(tl.float16)

    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = sm_h.to(tl.float32) + b
    y = tl.maximum(y, 0.0)

    tl.store(Out_ptr + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _softmax_bias_relu_kernel[(m,)](
            x, self.b2, out,
            n,
            x.stride(0),
            out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
