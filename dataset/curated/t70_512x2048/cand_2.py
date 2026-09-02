import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 70
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_gelu2_scale_softmax(X_ptr, Y_ptr, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # first gelu (exact erf), rounded back to bf16 like PyTorch
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # second gelu
    g = g * 0.5 * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale, rounded to bf16
    g = (g * 1.1999).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0

        if not h.is_cuda:
            h = F.gelu(h)
            h = F.gelu(h)
            h = h * 1.1999
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu2_scale_softmax[(Mrows,)](
            h, out, N, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
