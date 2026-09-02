import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 327
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_bias_scale_softmax(
    X, B1, B3, Out,
    D, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)

    # Emulate fp16 rounding at each elementwise step (matches eager half ops)
    y = (x + b1).to(tl.float16).to(tl.float32)
    y = (y * scale).to(tl.float16).to(tl.float32)
    y = (y + b3).to(tl.float16).to(tl.float32)

    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, d = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_scale_softmax[(m,)](
            y, self.b1, self.b3, out,
            d, 1.1132,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
