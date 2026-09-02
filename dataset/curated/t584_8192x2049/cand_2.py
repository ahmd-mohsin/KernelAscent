import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 584
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _gelu_double_rmsnorm_kernel(
    X, W1, W2, Y,
    D,
    stride_x, stride_y,
    SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), computed in fp32, rounded back to bf16 like PyTorch does
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm 1
    s1 = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / D
    r1 = tl.math.rsqrt(s1 + 1e-6)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = (g * r1).to(tl.bfloat16).to(tl.float32)
    y1 = (y1 * w1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    s2 = tl.sum(tl.where(mask, y1 * y1, 0.0), axis=0) / D
    r2 = tl.math.rsqrt(s2 + 1e-6)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = (y1 * r2).to(tl.bfloat16).to(tl.float32)
    y2 = (y2 * w2).to(tl.bfloat16).to(tl.float32)

    out = (y2 * SCALE).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xg = F.gelu(x)
            _xf = xg.float()
            xg = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = xg.float()
            xg = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return xg * 1.3399

        x = x.contiguous()
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.view(-1, d)
        m = x2.shape[0]

        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _gelu_double_rmsnorm_kernel[(m,)](
            x2, self.rms1_w, self.rms2_w, y,
            d,
            x2.stride(0), y.stride(0),
            1.3399,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
