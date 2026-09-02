import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 529
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _bias_rms_scale_kernel(
    X, B, W, Y,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add: bf16 + bf16 computed in fp32, rounded back to bf16 (matches PyTorch)
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)

    y1 = (xf * r).to(tl.bfloat16)  # (_xf * rsqrt).to(bf16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y2 = (y1.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)  # * rms2_w
    y3 = (y2.to(tl.float32) * scale).to(tl.bfloat16)  # * 1.181

    tl.store(Y + row * stride_y + cols, y3, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _bias_rms_scale_kernel[(M_,)](
            x, self.b1, self.rms2_w, y,
            N, x.stride(0), y.stride(0),
            1e-6, 1.181,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
