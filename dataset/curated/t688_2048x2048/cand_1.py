import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 688
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_act_softmax(Y, OUT, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    ptr = Y + row * stride + cols

    # load fp16 matmul result
    y = tl.load(ptr, mask=mask, other=0.0)

    # scale in fp16 (match reference fp16 arithmetic)
    scale = tl.full((1,), 1.0642, dtype=tl.float16)
    y = y * scale

    # relu (exact in fp16)
    zero16 = tl.zeros((BLOCK,), dtype=tl.float16)
    y = tl.maximum(y, zero16)

    # gelu: compute in fp32, round back to fp16 (matches PyTorch half gelu)
    xf = y.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax with float accumulation)
    z = g16.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT + row * stride + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_act_softmax[(Mrows,)](
            y, out, N, y.stride(0), BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )
        return out
