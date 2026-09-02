import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 284
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b2_ptr, w_ptr, out_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact, erf) computed in fp32, cast to bf16 like torch
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale
    y = (g * 1.164).to(tl.bfloat16).to(tl.float32)

    # bias add
    b = tl.load(b2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y + b).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    n = (z * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (n * w).to(tl.bfloat16)
    tl.store(out_ptr + row * D + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.b2, self.rms3_w, out, Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
