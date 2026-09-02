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
    X_ptr, W0_ptr, W1_ptr, Y_ptr,
    D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # --- first RMSNorm ---
    ms0 = tl.sum(x * x, axis=0) / D
    r0 = tl.math.rsqrt(ms0 + EPS)
    x1 = (x * r0).to(tl.bfloat16)                       # round to bf16 (matches .to(x.dtype))
    w0 = tl.load(W0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x1.to(tl.float32) * w0).to(tl.bfloat16)        # bf16 * bf16 elementwise (single rounding)

    # --- second RMSNorm ---
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms1 = tl.sum(yf * yf, axis=0) / D
    r1 = tl.math.rsqrt(ms1 + EPS)
    y1 = (yf * r1).to(tl.bfloat16)
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y1.to(tl.float32) * w1).to(tl.bfloat16)

    tl.store(Y_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.bfloat16:
            x = x.contiguous()
            rows = x.numel() // x.shape[-1]
            d = x.shape[-1]
            y = torch.empty_like(x)
            BLOCK = triton.next_power_of_2(d)
            _double_rmsnorm_kernel[(rows,)](
                x, self.rms0_w, self.rms1_w, y,
                D=d, EPS=1e-6, BLOCK=BLOCK,
                num_warps=8,
            )
            return y
        # fallback (CPU / other dtypes): reference implementation
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        return x
