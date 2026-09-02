import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 614
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_bias_scale_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # emulate fp16 rounding of (x + b) then (* scale), like eager PyTorch
    t = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
    t = (t.to(tl.float32) * scale).to(tl.float16)

    v = tl.where(mask, t.to(tl.float32), float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            t = (x + self.b0) * 1.4684
            return torch.softmax(t, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_bias_scale_softmax[(m,)](
            x2, self.b0, y,
            x2.stride(0), y.stride(0),
            n, 1.4684,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
