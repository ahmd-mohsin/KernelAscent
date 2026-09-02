import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 496
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_scale_rmsnorm_kernel(
    X, W, Y,
    D_: tl.constexpr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # replicate: x = x * 1.2952 (bf16 out), then x = x * 1.4173 (bf16 out)
    xf = x.to(tl.float32) * 1.2952
    xb = xf.to(tl.bfloat16)
    xf = xb.to(tl.float32) * 1.4173
    xb = xf.to(tl.bfloat16)

    # rmsnorm in fp32
    xf32 = xb.to(tl.float32)
    ms = tl.sum(xf32 * xf32, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)

    norm = (xf32 * r).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (norm.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D_)
        _fused_scale_rmsnorm_kernel[(M_,)](
            x, self.rms2_w, y,
            D_,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
