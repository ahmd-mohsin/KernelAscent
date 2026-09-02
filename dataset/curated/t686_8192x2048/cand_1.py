import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 686
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B1, G, B, W3, W4, B5, Y,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)  # fp16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)          # fp16
    x = x + b1                                             # fp16 add (matches torch)
    xf = x.to(tl.float32)

    # LayerNorm (fp32 internal, matches PyTorch half layer_norm)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * inv * g + b
    yh = y.to(tl.float16)

    # RMSNorm 1
    yf = yh.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)          # fp16
    yh = (yf * r).to(tl.float16) * w3                      # fp16 mul (matches torch)

    # RMSNorm 2
    yf = yh.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    yh = (yf * r).to(tl.float16) * w4

    b5 = tl.load(B5 + cols, mask=mask, other=0.0)
    yh = yh + b5

    tl.store(Y + row * N + cols, yh, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _fused_kernel[(Mrows,)](
            h, self.b1, self.ln2_g, self.ln2_b,
            self.rms3_w, self.rms4_w, self.b5, y,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y
