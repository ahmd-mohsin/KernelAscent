import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 204
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X, Y,
    N, stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # relu (in fp16, exact)
    x = tl.maximum(x, 0.0)

    # gelu (erf-based), computed in fp32, rounded back to fp16
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16)

    # scale, rounded to fp16
    s = (g.to(tl.float32) * 1.2304).to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    sf = s.to(tl.float32)
    sf = tl.where(mask, sf, float('-inf'))
    m = tl.max(sf, axis=0)
    e = tl.exp(sf - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_act_softmax_kernel[(Mrows,)](
            x, y, N, x.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
