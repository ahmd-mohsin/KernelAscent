import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 946
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_scale_ln_softmax(
    X, G, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # x * 1.316, rounded to bf16 like the reference (out-of-place bf16 mul)
    x = x * SCALE
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch's bf16 layer_norm)
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b

    # round to bf16 as layer_norm output would be, then softmax in fp32
    y = y.to(tl.bfloat16).to(tl.float32)

    m = tl.max(tl.where(mask, y, float('-inf')), axis=0)
    e = tl.where(mask, tl.exp(y - m), 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * D + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        x2 = x.view(-1, D_)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(D_)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_ln_softmax[(x2.shape[0],)](
            x2, self.ln1_g, self.ln1_b, out,
            D=D_, BLOCK=BLOCK, EPS=1e-5, SCALE=1.316,
            num_warps=num_warps,
        )
        return out.view(x.shape)
