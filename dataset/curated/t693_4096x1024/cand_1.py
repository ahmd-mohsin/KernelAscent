import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 693
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Out_ptr,
    n_cols,
    stride_xm, stride_om,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=-float('inf'))
    x = x.to(tl.float32) * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Out_ptr + row * stride_om + cols, y.to(Out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # bias add (elementwise) then cuBLAS matmul
        x = x + self.b0
        y = x @ self.W1
        # fused scale + softmax in one Triton kernel
        out = torch.empty_like(y)
        n_rows, n_cols = y.shape
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _scale_softmax_kernel[(n_rows,)](
            y, out,
            n_cols,
            y.stride(0), out.stride(0),
            1.2082,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
