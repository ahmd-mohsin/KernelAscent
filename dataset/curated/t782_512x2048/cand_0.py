import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 782
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_act_rms_softmax(
    X, W, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # exact gelu (erf), computed in fp32, rounded to bf16 like PyTorch output
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # rmsnorm in fp32
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.bfloat16)
    y = ((g * inv).to(tl.bfloat16) * w).to(tl.float32)

    # softmax (fp32 accumulation)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_act_rms_softmax[(m,)](
            x, self.rms3_w, out,
            x.stride(0), out.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out @ self.W5
