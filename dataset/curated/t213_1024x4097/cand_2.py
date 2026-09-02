import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 213
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _rms_gelu_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # match reference: cast to fp16 after rsqrt, then fp16 muls
    h = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = h * w
    h = h * scale.to(tl.float16)

    # exact erf-based gelu, computed in fp32 (opmath), cast back to fp16
    hf = h.to(tl.float32)
    g = hf * 0.5 * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    tl.store(Y + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _rms_gelu_kernel[(M_,)](
            x, self.rms1_w, y,
            N_, x.stride(0), y.stride(0),
            1e-6, 1.2985,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
