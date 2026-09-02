import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 440
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _ln_softmax_scale_kernel(
    X_ptr, G_ptr, B_ptr, OUT_ptr,
    stride_xm, stride_om,
    N, eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # match reference: layer_norm output is bf16 before softmax
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax
    y = tl.where(mask, y, float('-inf'))
    ymax = tl.max(y, axis=0)
    e = tl.exp(y - ymax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s) * scale

    tl.store(OUT_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        shape = h.shape
        h2 = h.reshape(-1, shape[-1])
        Mrows, N = h2.shape
        out = torch.empty_like(h2)
        BLOCK_N = triton.next_power_of_2(N)
        _ln_softmax_scale_kernel[(Mrows,)](
            h2, self.ln1_g, self.ln1_b, out,
            h2.stride(0), out.stride(0),
            N, 1e-5, 1.1165,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out.reshape(shape)
