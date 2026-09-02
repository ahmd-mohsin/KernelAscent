import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 365
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_kernel(
    X, B1, G, B, OUT,
    n_cols,
    stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.0149  (fp16 rounding to match reference)
    h = (x * 1.0149).to(tl.float16).to(tl.float32)
    # x = x + b1
    h = (h + b1).to(tl.float16).to(tl.float32)

    # softmax 1 (fp32 internal, fp16 output like PyTorch)
    hm = tl.where(mask, h, float('-inf'))
    m1 = tl.max(hm, axis=0)
    e1 = tl.exp(hm - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16).to(tl.float32)

    # layernorm
    mean = tl.sum(tl.where(mask, p1, 0.0), axis=0) / n_cols
    diff = tl.where(mask, p1 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd * g + bb).to(tl.float16).to(tl.float32)

    # softmax 2
    ym = tl.where(mask, y, float('-inf'))
    m2 = tl.max(ym, axis=0)
    e2 = tl.exp(ym - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    tl.store(OUT + row * stride_o + offs, p2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            h = x * 1.0149
            h = h + self.b1
            h = torch.softmax(h, dim=-1)
            h = F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.softmax(h, dim=-1)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(n_rows,)](
            x2, self.b1, self.ln3_g, self.ln3_b, out,
            n_cols,
            x2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
