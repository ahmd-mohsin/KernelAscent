import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 360
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _fused_bias_rmsnorm_kernel(
    X, B1, W, OUT,
    D_dim, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_dim

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B1 + cols, mask=mask, other=0.0)                  # fp16

    # x * 1.0393  (PyTorch half op computes in fp32 opmath, casts back to fp16)
    x = (x.to(tl.float32) * 1.0393).to(tl.float16)
    # x + b1  (fp32 opmath, cast back to fp16)
    x = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    xf = x.to(tl.float32)
    xf_m = tl.where(mask, xf, 0.0)
    mean_sq = tl.sum(xf_m * xf_m, axis=0) / D_dim
    inv = tl.rsqrt(mean_sq + eps)

    y = (xf * inv).to(tl.float16)  # cast to fp16 first (matches .to(x.dtype))
    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    out = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dd = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dd)
        _fused_bias_rmsnorm_kernel[(Mrows,)](
            x, self.b1, self.rms2_w, out,
            Dd, x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
