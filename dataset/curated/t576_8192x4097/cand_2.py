import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 576
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X, Y, G, B,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 math, round to bf16 like PyTorch output) ----
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf), fp32 opmath, round to bf16 ----
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- layernorm (fp32 stats over bf16 values) ----
    xz = tl.where(mask, x, 0.0)
    mean = tl.sum(xz, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- scale (fp32 opmath, round to bf16) ----
    x = x * 1.3939
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- softmax #2 ----
    x = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x, 0)
    e2 = tl.exp(x - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    y = e2 / s2

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.reshape(-1, N)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_row_kernel[(Mrows,)](
            x2d, y, self.ln2_g, self.ln2_b,
            N, x2d.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
