import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 415
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_scale_gelu_scale_softmax(
    X, Y,
    stride_xm, stride_ym,
    N,
    S1, S2,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # x = x * 1.0319  (fp16 op with fp32 opmath, result rounded to fp16)
    t = (x.to(tl.float32) * S1).to(tl.float16)

    # gelu (exact, erf-based), fp32 opmath, rounded to fp16
    tf = t.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = (0.5 * tf * (1.0 + tl.math.erf(tf * INV_SQRT2))).to(tl.float16)

    # x = x * 1.3895
    z = (g.to(tl.float32) * S2).to(tl.float16)

    # softmax in fp32 accumulation
    zf = tl.where(mask, z.to(tl.float32), float('-inf'))
    m = tl.max(zf, axis=0)
    e = tl.exp(zf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            t = x * 1.0319
            t = F.gelu(t)
            t = t * 1.3895
            return torch.softmax(t, dim=-1)

        x = x.contiguous()
        Mrows, N = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2d = x.view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _fused_scale_gelu_scale_softmax[(rows,)](
            x2d, y,
            x2d.stride(0), y.stride(0),
            N,
            1.0319, 1.3895,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view_as(x)
