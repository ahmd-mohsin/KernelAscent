import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 462
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_relu_rms_relu(X, W, Y, N, stride_x, stride_y, eps,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu in input dtype (bf16), then upcast to fp32
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # mean of squares in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    # normalize in fp32, round to bf16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 with fp32 opmath, rounded back to bf16
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_rms_relu[(m,)](
            x, self.rms2_w, y, n,
            x.stride(0), y.stride(0), 1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
