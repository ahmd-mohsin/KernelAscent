import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 203
M, D, DT = 512, 512, torch.float16


@triton.jit
def _rms_kernel(X, W, Y, N, stride_x, stride_y, scale, eps,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + eps)

    # match reference: cast normalized value to fp16 first
    xn16 = (xf * rrms).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # fp16*fp16 in PyTorch CUDA uses fp32 opmath then rounds to fp16
    z16 = (xn16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    # fp16 * python-float scalar: fp32 opmath, round to fp16
    out = (z16.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _rms_kernel[(Mrows,)](
            x2, self.rms0_w, y, N,
            x2.stride(0), y.stride(0),
            1.3616, 1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
