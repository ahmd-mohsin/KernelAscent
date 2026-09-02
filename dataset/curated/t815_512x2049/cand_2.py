import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 815
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _rms_scale_relu_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    s1, s2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # normalized, rounded to bf16 (matches .to(x.dtype))
    y = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> compute in fp32, round to bf16 (matches torch bf16 mul)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    # scalar muls, each rounded to bf16
    y = (y.to(tl.float32) * s1).to(tl.bfloat16)
    y = (y.to(tl.float32) * s2).to(tl.bfloat16)
    # relu
    y = tl.maximum(y, 0.0).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_scale_relu_kernel[(m,)](
            x, self.rms1_w, y,
            n, x.stride(0), y.stride(0),
            1e-6,
            1.3892, 1.0246,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
