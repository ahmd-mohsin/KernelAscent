import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 14
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_relu_rms2_relu(X, W1, W2, Y, D, stride_x, stride_y, eps,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)

    # RMSNorm 1 (fp32 accumulation, matches PyTorch)
    ms1 = tl.sum(x * x, axis=0) / D
    inv1 = tl.math.rsqrt(ms1 + eps)
    y = (x * inv1).to(tl.bfloat16)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    y = y * w1  # bf16 multiply (rounds like PyTorch)

    # RMSNorm 2
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / D
    inv2 = tl.math.rsqrt(ms2 + eps)
    z = (yf * inv2).to(tl.bfloat16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    z = z * w2

    # relu (exact for bf16 either way)
    zf = z.to(tl.float32)
    z = tl.maximum(zf, 0.0).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference fallback
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return torch.relu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_rms2_relu[(m,)](
            x2, self.rms1_w, self.rms2_w, out,
            d, x2.stride(0), out.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
