import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 189
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_ln_relu_softmax_gelu(
    X, G, B, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, like PyTorch's fp16 layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # emulate cast back to fp16 that happens between ops in eager mode
    y = y.to(tl.float16).to(tl.float32)

    # ---- ReLU ----
    y = tl.maximum(y, 0.0)

    # ---- Softmax (fp32 accumulation, fp16 output) ----
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # ---- scale (opmath fp32, cast back to fp16) ----
    p = p * SCALE
    p = p.to(tl.float16).to(tl.float32)

    # ---- GELU (exact, erf) ----
    out = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = torch.relu(y)
            y = torch.softmax(y, dim=-1)
            y = y * 1.4379
            return F.gelu(y)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_ln_relu_softmax_gelu[(rows,)](
            x2, self.ln0_g, self.ln0_b, y,
            N, x2.stride(0), y.stride(0),
            EPS=1e-5,
            SCALE=1.4379,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
