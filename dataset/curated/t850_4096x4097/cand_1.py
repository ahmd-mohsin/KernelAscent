import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 850
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _relu_scale_softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # relu in bf16
    x = tl.maximum(x, 0.0)
    # multiply by scalar, round to bf16 (matches torch bf16 elementwise semantics)
    x = (x.to(tl.float32) * SCALE).to(tl.bfloat16)
    # softmax computed in fp32 (matches torch softmax for bf16 input)
    xf = tl.where(mask, x.to(tl.float32), float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y_ptr + row * stride_ym + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _relu_scale_softmax_kernel[(m,)](
            h, out,
            n,
            h.stride(0), out.stride(0),
            SCALE=1.2179,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
