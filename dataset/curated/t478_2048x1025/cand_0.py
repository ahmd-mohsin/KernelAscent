import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 478
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_relu_rms_relu(X, W, Y, N, stride_x, stride_y, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu (applied twice in ref == once)
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + 1e-6)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xf * rs).to(tl.float16) * w
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_relu_rms_relu[(m,)](
            h, self.rms3_w, out, n,
            h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
