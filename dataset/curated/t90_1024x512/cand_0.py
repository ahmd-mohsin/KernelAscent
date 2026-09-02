import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 90
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, B, Y, N, eps, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.bfloat16)  # round to bf16 like reference .to(x.dtype)

    # multiply by rms weight (bf16 op, fp32 compute, round to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    h = (xn.to(tl.float32) * w).to(tl.bfloat16)

    # exact GELU (erf-based) in fp32, round to bf16
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # add bias, round to bf16
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    o = (g.to(tl.float32) + b).to(tl.bfloat16)

    # scale, round to bf16
    o = (o.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(Y + row * N + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        x2 = x.view(-1, N)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(x2.shape[0],)](
            x2, self.rms0_w, self.b2, y, N, 1e-6, 1.0205,
            BLOCK=BLOCK, num_warps=4,
        )
        return y.view_as(x)
