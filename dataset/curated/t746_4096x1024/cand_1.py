import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 746
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    # scale (opmath fp32, round back to bf16 like PyTorch elementwise)
    x = (x.to(tl.float32) * 1.4535).to(X.dtype.element_ty)
    # relu
    x = tl.maximum(x, 0.0)
    # exact gelu in fp32 (PyTorch opmath), round back to bf16
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(X.dtype.element_ty).to(tl.float32)
    g = tl.where(mask, g, float('-inf'))
    # softmax in fp32
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.4535
            x = torch.relu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
        _fused_kernel[(n_rows,)](
            x2, y, n_cols, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
