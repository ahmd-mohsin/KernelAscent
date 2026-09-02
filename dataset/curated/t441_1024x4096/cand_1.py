import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 441
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X, B1, B2, OUT,
    stride_xm, stride_om,
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, like PyTorch's half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = e1 / s1

    # cast to fp16 (output dtype of first softmax), then fp16 adds like reference
    y16 = y.to(tl.float16)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    y16 = y16 + b1
    y16 = y16 + b2

    # softmax 2 in fp32
    z = y16.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    m2 = tl.max(z, axis=0)
    e2 = tl.exp(z - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2) * scale

    tl.store(OUT + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N_ = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(M_,)](
            x, self.b1, self.b2, out,
            x.stride(0), out.stride(0),
            N_, 1.2172,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
