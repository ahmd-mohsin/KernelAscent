import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 533
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_relu_rms_scale_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu in bf16
    zero = tl.zeros_like(x)
    x = tl.maximum(x, zero)

    # float32 compute
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    rs = tl.math.rsqrt(ms + 1e-6)

    # normalized, cast back to bf16 (matches .to(x.dtype))
    y = (xf * rs).to(tl.bfloat16)

    # * weight : bf16*bf16 done in fp32 then rounded to bf16 (PyTorch opmath)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # sequential scalar multiplies, each rounded to bf16 like PyTorch
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
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_relu_rms_scale_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
