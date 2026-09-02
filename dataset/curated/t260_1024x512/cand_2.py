import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 260
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_relu_bias_rmsnorm_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # relu in bf16 (exact), then add in fp32 rounded back to bf16 (matches torch bf16 add)
    xr = tl.maximum(x, 0.0)
    xb = (xr.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    y = (xf * inv).to(tl.bfloat16)  # cast to bf16 first (matches reference)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_bias_rmsnorm_kernel[(m,)](
            x, self.b1, self.rms2_w, out,
            x.stride(0),
            N=n, EPS=1e-6, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
