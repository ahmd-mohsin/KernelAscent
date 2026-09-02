import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 616
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_gelu_softmax_relu_ln(
    X, G, B, Y,
    D_size,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_size

    x = tl.load(X + row * D_size + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based) computed in fp32, rounded to fp16 like PyTorch op output
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax with fp32 accumulation, fp16 output (matches PyTorch half softmax)
    gm = tl.where(mask, g, float('-inf'))
    m = tl.max(gm, 0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # ReLU
    p = tl.maximum(p, 0.0)

    # LayerNorm in fp32
    mean = tl.sum(p, 0) / D_size
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, 0) / D_size
    rstd = 1.0 / tl.sqrt(var + eps)

    gw = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    bw = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * gw + bw

    tl.store(Y + row * D_size + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_gelu_softmax_relu_ln[(rows,)](
            x2, self.ln3_g, self.ln3_b, y,
            d, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
