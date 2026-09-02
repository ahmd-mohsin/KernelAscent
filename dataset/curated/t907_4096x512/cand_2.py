import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 907
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_relu_scale_rmsnorm(
    X, W, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # bf16

    # relu in bf16 (exact)
    x = tl.maximum(x, 0.0)

    # x * 1.0822 : compute in fp32, round to bf16 (matches PyTorch opmath)
    xf = x.to(tl.float32) * 1.0822
    x = xf.to(tl.bfloat16)

    # x * 1.1211 : compute in fp32, round to bf16
    xf = x.to(tl.float32) * 1.1211
    x = xf.to(tl.bfloat16)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.bfloat16)

    # multiply by weight: fp32 opmath then round to bf16
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_relu_scale_rmsnorm[(m,)](
            x2, self.rms3_w, y,
            x2.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
