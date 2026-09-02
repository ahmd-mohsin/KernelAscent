import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 883
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _fused_kernel(
    X, Y, G, B,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1

    # softmax 2 (mask out-of-bounds as -inf)
    p1m = tl.where(mask, p1, float('-inf'))
    m2 = tl.max(p1m, axis=0)
    e2 = tl.exp(p1m - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    # layer norm
    mean = tl.sum(tl.where(mask, p2, 0.0), axis=0) / N
    diff = tl.where(mask, p2 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (p2 - mean) * rstd * g + b
    y = y * scale

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x * 1.0249

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(rows,)](
            x2, y, self.ln2_g, self.ln2_b,
            N, x2.stride(0), y.stride(0),
            1e-5, 1.0249,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
