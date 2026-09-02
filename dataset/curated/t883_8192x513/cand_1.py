import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 883
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _fused_softmax2_ln_kernel(
    X, W, B, Y,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x_ptr = X + row * stride_x + offs
    x = tl.load(x_ptr, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accum, round to fp16 like PyTorch output) ----
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- softmax #2 ----
    y1m = tl.where(mask, y1, float('-inf'))
    m2 = tl.max(y1m, axis=0)
    e2 = tl.exp(y1m - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = (e2 / s2).to(tl.float16).to(tl.float32)

    # ---- layernorm (fp32 stats) ----
    mean = tl.sum(tl.where(mask, y2, 0.0), axis=0) / N
    diff = tl.where(mask, y2 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    out = (y2 - mean) * rstd * w + b
    out = out.to(tl.float16).to(tl.float32) * scale

    tl.store(Y + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = torch.softmax(y, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return y * 1.0249

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_softmax2_ln_kernel[(rows,)](
            x2, self.ln2_g, self.ln2_b, out,
            N, x2.stride(0), out.stride(0),
            1e-5, 1.0249,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
