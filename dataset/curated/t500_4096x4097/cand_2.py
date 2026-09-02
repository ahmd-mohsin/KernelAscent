import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 500
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_bias_ln_kernel(
    X, B1, G, B2, OUT,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptr = X + row * stride_x + cols
    x = tl.load(x_ptr, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, like PyTorch bf16 softmax) ----
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # round softmax result to bf16 (matches reference intermediate dtype)
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- bias add, rounded to bf16 (matches reference intermediate) ----
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (p + b1).to(tl.bfloat16).to(tl.float32)
    y = tl.where(mask, y, 0.0)

    # ---- layernorm (fp32 accumulation) ----
    n_f = N.to(tl.float32)
    mean = tl.sum(y, axis=0) / n_f
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y - mean) * rstd * g + b2

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 16 if BLOCK >= 8192 else 8

        _fused_softmax_bias_ln_kernel[(m,)](
            x2, self.b1, self.ln2_g, self.ln2_b, out,
            n, x2.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
