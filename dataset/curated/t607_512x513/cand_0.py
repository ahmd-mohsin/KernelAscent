import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 607
M, D, DT = 512, 513, torch.float16


@triton.jit
def _ln_fwd_kernel(X, G, B, Y, N, stride_x, stride_y, eps,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


@triton.jit
def _rms_fwd_kernel(X, W, Y, N, stride_x, stride_y, eps,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + eps)
    # match reference: normalize in fp32 -> cast to fp16 -> multiply by w
    # (torch half mul uses fp32 opmath, so recompute product in fp32 then round)
    xn = (x * r).to(tl.float16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = xn * w
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mr, N = x.shape

        # fused LayerNorm (fp32 accumulation, matches PyTorch half layer_norm)
        y = torch.empty_like(x)
        _ln_fwd_kernel[(Mr,)](
            x, self.ln0_g, self.ln0_b, y,
            N, x.stride(0), y.stride(0), 1e-5,
            BLOCK=triton.next_power_of_2(N),
            num_warps=4,
        )

        # matmul via cuBLAS tensor cores (same as reference @)
        z = y @ self.W1

        # fused RMSNorm + weight
        N2 = z.shape[1]
        out = torch.empty_like(z)
        _rms_fwd_kernel[(Mr,)](
            z, self.rms2_w, out,
            N2, z.stride(0), out.stride(0), 1e-6,
            BLOCK=triton.next_power_of_2(N2),
            num_warps=4,
        )
        return out
