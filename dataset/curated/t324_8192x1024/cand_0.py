import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 324
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_bias_relu_softmax(
    X, B, Y,
    n_cols,
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add + relu in input dtype (matches eager bf16 rounding)
    y = x + b
    zero = tl.zeros_like(y)
    y = tl.where(y > zero, y, zero)

    # softmax in fp32 (matches PyTorch internal accumulation)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.relu(x + self.b0)
            return torch.softmax(y, dim=-1)

        x2 = x.contiguous()
        orig_shape = x2.shape
        n_cols = orig_shape[-1]
        x2 = x2.view(-1, n_cols)
        n_rows = x2.shape[0]

        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_bias_relu_softmax[(n_rows,)](
            x2, self.b0, out,
            n_cols,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
