import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 219
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _softmax_scale_kernel(X, Y, N, stride_x, stride_y, SCALE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)
    scale = tl.full((1,), SCALE, dtype=tl.float16)
    y = y * scale
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h += self.b1
        z = h @ self.W2
        out = torch.empty_like(z)
        Mrows, N = z.shape
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_scale_kernel[(Mrows,)](
            z, out, N, z.stride(0), out.stride(0), 1.481,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
