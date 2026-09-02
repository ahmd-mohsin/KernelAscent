import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 533
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _relu_rmsnorm_scale_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    D_: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu (in input dtype, exact)
    x = tl.where(x > 0, x, 0.0 * x)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    rstd = 1.0 / tl.sqrt(ms + EPS)

    # normalize in fp32, round to bf16 (matches .to(x.dtype))
    y = (xf * rstd).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 in PyTorch: computed in fp32, rounded back to bf16
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    # scalar muls: each computed in fp32, rounded back to bf16
    y = (y.to(tl.float32) * 1.1764).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.103).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.281).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(d)
        _relu_rmsnorm_scale_kernel[(m,)](
            x2d, self.rms1_w, y,
            x2d.stride(0), y.stride(0),
            d, 1e-6, BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
