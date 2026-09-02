import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 349
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W, B, Y,
    n_cols,
    stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 accumulate, round to bf16 like PyTorch output) ----
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p = e1 / s1
    p = p.to(tl.bfloat16).to(tl.float32)  # intermediate bf16 rounding

    # ---- layer norm (fp32 math, bf16 output rounding) ----
    n = n_cols.to(tl.float32)
    mean = tl.sum(tl.where(mask, p, 0.0), axis=0) / n
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b
    y = y.to(tl.bfloat16).to(tl.float32)  # intermediate bf16 rounding

    # ---- relu ----
    y = tl.maximum(y, 0.0)

    # ---- softmax 2 ----
    y = tl.where(mask, y, float('-inf'))
    m2 = tl.max(y, axis=0)
    e2 = tl.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_kernel[(n_rows,)](
            x2d, self.ln1_g, self.ln1_b, out,
            n_cols,
            x2d.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
