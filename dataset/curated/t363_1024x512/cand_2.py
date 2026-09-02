import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 363
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = x * rstd
    # cast to bf16 then multiply by weight (match reference: .to(dtype) * w)
    xn_bf = xn.to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = xn_bf * w
    # exact gelu computed as F.gelu does on bf16 input (internally fp32-ish);
    # F.gelu on bf16: computes in bf16? PyTorch computes elementwise in fp32 then casts.
    hf = h.to(tl.float32)
    g = hf * 0.5 * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    tl.store(Y + row * N + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _rms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, y, N, 1e-6,
            BLOCK_N=4096, num_warps=8,
        )
        return y
