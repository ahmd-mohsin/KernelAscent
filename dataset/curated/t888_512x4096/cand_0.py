import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 888
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_gelu3_rms_kernel(
    X, W, Y,
    N,
    stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.352 (computed in fp32, stored bf16)
    x = x * 1.352
    x = x.to(tl.bfloat16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf, fp32 opmath, round to bf16)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)
    # gelu #2
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)
    # gelu #3
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    xm = tl.where(mask, x, 0.0)
    ms = tl.sum(xm * xm, axis=0) / N
    inv_rms = tl.math.rsqrt(ms + eps)
    xn = (x * inv_rms).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.352
            x = F.gelu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_gelu3_rms_kernel[(rows,)](
            x2, self.rms4_w, y,
            N,
            x2.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
