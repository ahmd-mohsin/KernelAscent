import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 136
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # Elementwise ops matching bf16 rounding of the reference
    x = (x.to(tl.float32) * 1.0831).to(tl.bfloat16)
    x = (x.to(tl.float32) * 1.4976).to(tl.bfloat16)
    x = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # Softmax in fp32 (as PyTorch does for bf16 inputs), then round to bf16
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # GELU (exact, erf-based) computed in fp32 on bf16 values, output bf16
    v = sm.to(tl.float32)
    g = v * 0.5 * (1.0 + tl.math.erf(v * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x2, self.b2, y,
            x2.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
