import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 698
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_gelu_rms_gelu(X, W, Y, D: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # gelu #1 (exact, fp32 opmath, round to bf16 like PyTorch)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / D
    r = tl.math.rsqrt(ms + eps)
    n = (g * r).to(tl.bfloat16).to(tl.float32)

    # scale by weight (bf16 mul with fp32 opmath, round to bf16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    h = (n * w).to(tl.bfloat16).to(tl.float32)

    # gelu #2
    o = h * 0.5 * (1.0 + tl.math.erf(h * 0.7071067811865476))

    tl.store(Y + row * D + offs, o.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_rms_gelu[(m,)](
            x2, self.rms1_w, y, d, 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
