import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 419
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_bias_rms_softmax(
    X, B0, W1, OUT,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0  (bf16 elementwise: fp32 math, round to bf16)
    y = (x + b).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)

    n = (y * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch bf16 softmax)
    z = tl.where(mask, z, float("-inf"))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        buf = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_bias_rms_softmax[(rows,)](
            x, self.b0, self.rms1_w, buf,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return buf @ self.W3
