import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 630
M, D, DT = 4096, 1024, torch.float16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _gelu_f32(x):
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def _fused_gelu_softmax_gelu2(
    X, Y,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # gelu #1 (compute fp32, round to fp16 like PyTorch half kernel)
    g = _gelu_f32(x)
    g = g.to(tl.float16).to(tl.float32)

    # softmax over the row (fp32 accumulation, fp16-rounded output)
    g_masked = tl.where(mask, g, float("-inf"))
    m = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # gelu #2
    p = _gelu_f32(p)
    p = p.to(tl.float16).to(tl.float32)

    # gelu #3
    p = _gelu_f32(p)

    tl.store(Y + row * stride_y + offs, p.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores on A100)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_gelu_softmax_gelu2[(Mrows,)](
            h, out, N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
