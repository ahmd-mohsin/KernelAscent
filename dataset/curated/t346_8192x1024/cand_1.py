import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 346
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_kernel(X, B, Y, stride_x, stride_y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf) computed in fp32, rounded to fp16 like PyTorch
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax with fp32 accumulation
    g_masked = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = (e / denom).to(tl.float16)

    # ReLU (no-op for softmax outputs, but keep for exactness)
    s = tl.maximum(s, 0.0)

    # Bias add in fp32 opmath, round to fp16
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (s.to(tl.float32) + b).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = torch.softmax(y, dim=-1)
            y = torch.relu(y)
            return y + self.b3

        x = x.contiguous()
        orig_shape = x.shape
        Dd = orig_shape[-1]
        x2 = x.view(-1, Dd)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dd)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(rows,)](
            x2, self.b3, y,
            x2.stride(0), y.stride(0),
            Dd, BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
