import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 488
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, OUT,
    G1, B1, G3, B3,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # scale (matches bf16 rounding of elementwise op)
    x = x * SCALE
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (fp32 accumulation, bf16 output rounding)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation, bf16 output rounding)
    y_masked = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 3
    mean2 = tl.sum(tl.where(mask, p, 0.0), axis=0) / N
    pm = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(pm * pm, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = pm * rstd2 * g3 + b3

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x * 1.2406
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.softmax(y, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _fused_kernel[(rows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            N, x2.stride(0), out.stride(0),
            SCALE=1.2406,
            EPS=1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
