import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 72
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, out_ptr,
    g1_ptr, b1_ptr, g3_ptr, b3_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (fp32 compute, round to bf16 like PyTorch op boundary) ----
    xmax = tl.max(x, axis=0)
    ex = tl.exp(x - xmax)
    ex = tl.where(mask, ex, 0.0)
    s = tl.sum(ex, axis=0)
    y = ex / s
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- layer norm 1 ----
    n = N.to(tl.float32)
    mean = tl.sum(y, axis=0) / n
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(g1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y - mean) * rstd * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- relu ----
    y = tl.maximum(y, 0.0)

    # ---- layer norm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / n
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(g3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y - mean2) * rstd2 * g3 + b3

    tl.store(out_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.relu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(rows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
