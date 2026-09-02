import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 810
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_act_softmax(X, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load fp16 result of the matmul
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)

    # scale in fp16 (matches PyTorch fp16 tensor * scalar)
    scale = tl.full((1,), 1.2221, dtype=tl.float16)
    x = (x * scale).to(tl.float16)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # first exact GELU: compute in fp32 (PyTorch opmath), cast back to fp16
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = g.to(tl.float16)

    # second exact GELU
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = g.to(tl.float16)

    # relu (exact in fp16)
    zero = tl.full((1,), 0.0, dtype=tl.float16)
    x = tl.maximum(x, zero)

    # softmax with fp32 accumulation (matches PyTorch half softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS handles the fp16 GEMM optimally on A100 (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_act_softmax[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
