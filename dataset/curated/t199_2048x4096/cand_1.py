import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 199
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _double_rms_kernel(X, W1, W2, Y, N, stride_x, stride_y, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First RMSNorm
    ms1 = tl.sum(x * x, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + EPS)
    x1 = (x * r1).to(tl.bfloat16)  # cast to bf16 like reference
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> fp32 compute, round to bf16 (matches PyTorch semantics)
    x1 = (x1.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # Second RMSNorm
    xf = x1.to(tl.float32)
    ms2 = tl.sum(xf * xf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + EPS)
    x2 = (xf * r2).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    out = (x2.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _double_rms_kernel[(Mrows,)](
            x, self.rms1_w, self.rms2_w, y,
            N, x.stride(0), y.stride(0),
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
