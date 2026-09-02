import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 21
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (erf) in fp32, round to bf16 (matches PyTorch op boundary)
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, round to bf16
    g_m = tl.where(mask, g, float("-inf"))
    m1 = tl.max(g_m, axis=0)
    e1 = tl.exp(g_m - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    sm1 = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # gelu again
    g2 = sm1 * 0.5 * (1.0 + tl.math.erf(sm1 * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # add bias (bf16 add, matches PyTorch elementwise upcast->round)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    a = (g2 + b).to(tl.bfloat16).to(tl.float32)

    # final softmax
    a_m = tl.where(mask, a, float("-inf"))
    m2 = tl.max(a_m, axis=0)
    e2 = tl.exp(a_m - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            x = x + self.b3
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x2, self.b3, y, n,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
