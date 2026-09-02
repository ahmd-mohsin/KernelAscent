import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 624
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_act_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # relu
    xf = tl.maximum(xf, 0.0)
    # exact gelu (erf) computed in fp32 (matches PyTorch opmath for bf16)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    # relu again
    g = tl.maximum(g, 0.0)

    # cast to bf16 (as reference does before rms), then upcast to fp32
    gb = g.to(tl.bfloat16)
    gf = gb.to(tl.float32)

    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    normed = (gf * inv).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        BLOCK = triton.next_power_of_2(n)
        _fused_act_rms_kernel[(m,)](
            x, self.rms4_w, out,
            x.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
