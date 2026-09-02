import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 325
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(
    X, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # replicate fp16 elementwise scaling (round to fp16 after each mul,
    # matching the reference which operates on fp16 tensors)
    x = (x * 1.3389).to(tl.float16)
    x = (x * 1.4601).to(tl.float16)

    # exact GELU in fp32 math (PyTorch half GELU uses float opmath), then fp16
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g = g.to(tl.float16)

    # softmax with float32 accumulation (matches PyTorch half softmax)
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    row_max = tl.max(gf, axis=0)
    e = tl.exp(gf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_scale_gelu_softmax[(m,)](
            h, y,
            h.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
