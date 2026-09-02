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
    stride_xm, stride_ym,
    N, eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * scale

    # LayerNorm (fp32 accumulation, matching PyTorch internals)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # Round to bf16 (matching reference intermediate dtype), then softmax in fp32
    y = y.to(tl.bfloat16).to(tl.float32)

    y = tl.where(mask, y, float('-inf'))
    ymax = tl.max(y, axis=0)
    e = tl.exp(y - ymax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_scale_ln_softmax[(m,)](
            x, self.ln1_g, self.ln1_b, y,
            x.stride(0), y.stride(0),
            n, 1e-5, 1.316,
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        return y
