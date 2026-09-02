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
    D: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # First RMSNorm
    ms0 = tl.sum(xf * xf, axis=0) / D
    rs0 = 1.0 / tl.sqrt(ms0 + eps)
    y = (xf * rs0).to(tl.bfloat16)  # round to bf16 like reference

    w0 = tl.load(W0_ptr + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w0.to(tl.float32)).to(tl.bfloat16)

    # Second RMSNorm
    yf = y.to(tl.float32)
    ms1 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / D
    rs1 = 1.0 / tl.sqrt(ms1 + eps)
    z = (yf * rs1).to(tl.bfloat16)

    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y_ptr + row * D + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _double_rmsnorm_kernel[(m,)](
            x2, self.rms0_w, self.rms1_w, out,
            d, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
