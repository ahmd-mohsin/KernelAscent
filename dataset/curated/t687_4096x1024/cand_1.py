import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 687
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _gelu_rms_kernel(
    x_ptr, w_ptr, out_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 like PyTorch's opmath
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # cast to bf16 (matches x after F.gelu), then back to fp32 for RMS stats
    gb = g.to(tl.bfloat16)
    gf = gb.to(tl.float32)

    ms = tl.sum(gf * gf, axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)

    normed = (gf * inv).to(tl.bfloat16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # bf16 * bf16 in PyTorch computes in fp32 then rounds
    y = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * D + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 1024 else 4
        _gelu_rms_kernel[(m,)](
            x2, self.rms1_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
