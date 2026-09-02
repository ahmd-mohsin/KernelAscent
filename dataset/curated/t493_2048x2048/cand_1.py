import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 493
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 accumulate, round to bf16 like the eager op) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    p1 = e1 / s1
    p1 = p1.to(tl.bfloat16).to(tl.float32)

    # ---- softmax 2 ----
    p1m = tl.where(mask, p1, float('-inf'))
    m2 = tl.max(p1m, 0)
    e2 = tl.exp(p1m - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    p2 = e2 / s2
    p2 = p2.to(tl.bfloat16).to(tl.float32)

    # ---- scale (bf16 rounding to match eager x * 1.4655 on bf16) ----
    z = p2 * 1.4655
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU: 0.5 * z * (1 + erf(z / sqrt(2))) ----
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Y + row * stride_y + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2:
            # fallback (correctness path)
            y = torch.softmax(x, dim=-1)
            y = torch.softmax(y, dim=-1)
            y = y * 1.4655
            return F.gelu(y)

        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_kernel[(n_rows,)](
            x, y, n_cols, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
