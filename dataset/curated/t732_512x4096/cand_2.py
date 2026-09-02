import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 732
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _rmsnorm_scale_kernel(
    X, W, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + EPS)

    # match reference rounding: cast to fp16 after normalization
    xn = (x * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)

    # x_half * w_half computed in fp32, rounded to fp16 (PyTorch opmath)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    # then * 1.0862 in fp32, rounded to fp16
    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 4096 else 4

        _rmsnorm_scale_kernel[(m,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            D=d, EPS=1e-6, SCALE=1.0862,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
