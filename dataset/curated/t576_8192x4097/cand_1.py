import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 576
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_gelu_ln_softmax(
    X, Y, G, B,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 accumulate, round to bf16 like PyTorch) ----
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf), fp32 opmath, round to bf16 ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- layer norm (fp32 stats), round to bf16 ----
    x = tl.where(mask, x, 0.0)
    mean = tl.sum(x, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- scale (fp32 opmath, round to bf16) ----
    x = x * scale
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- softmax 2 ----
    x = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x, 0)
    e2 = tl.exp(x - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    y = e2 / s2

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: original path
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = x * 1.3939
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_gelu_ln_softmax[(rows,)](
            x2, y, self.ln2_g, self.ln2_b,
            N, x2.stride(0), y.stride(0),
            1e-5, 1.3939,
            BLOCK=BLOCK,
            num_warps=32,
            num_stages=1,
        )
        return y.view(orig_shape)
