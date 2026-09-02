import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 493
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_double_softmax_gelu(
    X, Y,
    N_COLS,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accumulate, round to bf16 like PyTorch output) ----
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1
    p1 = p1.to(tl.bfloat16).to(tl.float32)

    # ---- softmax #2 ----
    p1m = tl.where(mask, p1, -float('inf'))
    m2 = tl.max(p1m, axis=0)
    e2 = tl.exp(p1m - m2)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2
    p2 = p2.to(tl.bfloat16).to(tl.float32)

    # ---- scale (bf16 rounding to match separate op) ----
    z = p2 * 1.4655
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf) in fp32 ----
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Y + row * stride_y + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = torch.softmax(y, dim=-1)
            y = y * 1.4655
            return F.gelu(y)

        x = x.contiguous()
        n_rows, n_cols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        orig_shape = x.shape
        x2d = x.view(-1, n_cols)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_double_softmax_gelu[(rows,)](
            x2d, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
