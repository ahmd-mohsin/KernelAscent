import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 197
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _gelu_rms_kernel(X, W, Y, N, eps,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU (erf variant), computed in fp32 then cast to fp16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)
    ms = tl.sum(gf * gf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(W + offs, mask=mask, other=0.0)
    out = (gf * inv).to(tl.float16) * w
    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _gelu_rms_kernel[(Mrows,)](
            x, self.rms2_w, y, N, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
