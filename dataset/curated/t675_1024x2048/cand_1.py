import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 675
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w3_ptr, w4_ptr, out_ptr, N, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load bf16 -> fp32
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact, computed in fp32, rounded to bf16 like PyTorch)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # scale
    x = x * 1.113
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # rmsnorm #1
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)
    x = x * inv
    w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x.to(tl.bfloat16).to(tl.float32) * w3)
    x = x.to(tl.bfloat16).to(tl.float32)

    # rmsnorm #2
    ms2 = tl.sum(x * x, axis=0) / N
    inv2 = tl.math.rsqrt(ms2 + EPS)
    x = x * inv2
    w4 = tl.load(w4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x.to(tl.bfloat16).to(tl.float32) * w4)

    tl.store(out_ptr + row * N + cols, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x, self.rms3_w, self.rms4_w, out,
            n, 1e-6, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
