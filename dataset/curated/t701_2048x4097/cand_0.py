import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 701
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _fused_softmax_kernel(
    X, B, Y,
    n_cols,
    stride_xm, stride_ym,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # x + b0: half+half computed in fp32 then rounded to half (PyTorch opmath)
    v = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # softmax computed in fp32 accumulation, result rounded to half
    f = v.to(tl.float32)
    f = tl.where(mask, f, float('-inf'))
    row_max = tl.max(f, axis=0)
    e = tl.exp(f - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = (e / denom).to(tl.float16)

    # scale: half * float scalar computed in fp32, rounded to half
    out = (s.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK <= 4096 else 16
        _fused_softmax_kernel[(m,)](
            x, self.b0, y,
            n,
            x.stride(0), y.stride(0),
            1.2808,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
