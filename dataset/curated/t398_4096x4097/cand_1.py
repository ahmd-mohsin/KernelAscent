import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 398
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_bias_rms2_relu(X, B, W2, W3, Y, N, stride_x, stride_y, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp32, round to bf16 (matches PyTorch bf16 elementwise add)
    t = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 1
    xf = t.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)
    y = (xf * r).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 2
    xf = y.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)
    y = (xf * r).to(tl.bfloat16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16)

    # ReLU
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_rms2_relu[(m,)](
            x, self.b1, self.rms2_w, self.rms3_w, y,
            n, x.stride(0), y.stride(0),
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
