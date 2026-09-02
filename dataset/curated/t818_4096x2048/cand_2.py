import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 818
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X_ptr, W0_ptr, W1_ptr, Out_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # ---- RMSNorm 0 ----
    ms0 = tl.sum(xf * xf, axis=0) / D
    r0 = 1.0 / tl.sqrt(ms0 + EPS)
    y = (xf * r0).to(tl.bfloat16)  # round to bf16 as in reference

    w0 = tl.load(W0_ptr + cols, mask=mask, other=0.0)
    z = (y.to(tl.float32) * w0.to(tl.float32)).to(tl.bfloat16)  # bf16 mul (single rounding)

    # ---- RMSNorm 1 ----
    zf = z.to(tl.float32)
    ms1 = tl.sum(zf * zf, axis=0) / D
    r1 = 1.0 / tl.sqrt(ms1 + EPS)
    y1 = (zf * r1).to(tl.bfloat16)

    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)
    out = (y1.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out_ptr + row * D + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda and x.dtype == torch.bfloat16
        x = x.contiguous()
        Mrows, Dcols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _double_rmsnorm_kernel[(Mrows,)](
            x, self.rms0_w, self.rms1_w, out,
            D=Dcols, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
