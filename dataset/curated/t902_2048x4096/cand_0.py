import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 902
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_softmax2_gelu_kernel(
    X_ptr, Y_ptr,
    n_cols,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accumulation, like PyTorch on fp16 input) ----
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = e1 / tl.sum(e1, axis=0)

    # round to fp16 to match intermediate dtype of the reference
    s1 = s1.to(tl.float16).to(tl.float32)
    s1 = tl.where(mask, s1, float('-inf'))

    # ---- softmax #2 ----
    m2 = tl.max(s1, axis=0)
    e2 = tl.exp(s1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = e2 / tl.sum(e2, axis=0)

    # round to fp16 to match intermediate dtype of the reference
    t = s2.to(tl.float16).to(tl.float32)

    # ---- exact GELU: 0.5 * t * (1 + erf(t / sqrt(2))) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * t * (1.0 + tl.math.erf(t * INV_SQRT2))

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return F.gelu(x)

        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2d.shape

        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_softmax2_gelu_kernel[(n_rows,)](
            x2d, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
