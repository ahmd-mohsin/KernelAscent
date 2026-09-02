import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 791
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(
    X, Y,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # replicate the two fp16-rounded scalar multiplies
    x = (x * 1.3036).to(tl.float16).to(tl.float32)
    x = (x * 1.0726).to(tl.float16).to(tl.float32)

    # exact (erf) GELU in fp32, then round to fp16 like the reference
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch half-softmax accumulation)
    g = tl.where(mask, g, float("-inf"))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_scale_gelu_softmax[(m,)](
            h, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
