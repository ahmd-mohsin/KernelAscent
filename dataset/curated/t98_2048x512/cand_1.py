import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 98
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_kernel(X, W2, B4, W5, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)  # fp16

    # x = x * 1.1988 (in fp16, matching PyTorch fp16 elementwise)
    x = (x.to(tl.float32) * 1.1988).to(tl.float16)

    # RMSNorm 1 (stats in fp32)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    x = xh * w2  # fp16

    # ReLU
    x = tl.maximum(x, tl.zeros_like(x))

    # + b4
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    x = x + b4  # fp16

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)

    w5 = tl.load(W5 + cols, mask=mask, other=0.0)
    y = xh * w5

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # tensor-core GEMM
        x = x.contiguous()
        rows, n = x.shape
        y = torch.empty_like(x)
        _fused_kernel[(rows,)](
            x, self.rms2_w, self.b4, self.rms5_w, y, n,
            BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return y
