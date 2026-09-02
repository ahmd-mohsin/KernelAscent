import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 979
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, B1, W2, W3, G4, B4, Y,
    stride_x, stride_y,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    x = x + b1  # fp16 add

    # RMSNorm 1
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    x = (xf * inv).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    x = x * w2  # fp16 mul

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    x = (xf * inv).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    x = x * w3  # fp16 mul

    # LayerNorm (fp32 compute, like PyTorch for half input)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd) * g4 + b4
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.mm(x, self.W0)  # (M, 512) fp16, tensor cores
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _fused_norm_kernel[(Mrows,)](
            h, self.b1, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b, y,
            h.stride(0), y.stride(0),
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=4,
        )
        return y
