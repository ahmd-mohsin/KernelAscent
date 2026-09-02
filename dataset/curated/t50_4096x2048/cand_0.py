import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 50
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_kernel(X, Y, G1, B1, G3, B3,
                  stride_x, stride_y,
                  N, scale1, scale2, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32), then round to fp16 like the reference intermediate
    xmax = tl.max(x, axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom
    s = s.to(tl.float16).to(tl.float32)

    # layernorm 1
    mean1 = tl.sum(s, axis=0) / N
    d1 = tl.where(mask, s - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    inv1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = (d1 * inv1) * g1 + b1
    y1 = y1.to(tl.float16).to(tl.float32)

    # scale, rounded to fp16
    y1 = (y1 * scale1).to(tl.float16).to(tl.float32)

    # layernorm 2
    mean2 = tl.sum(tl.where(mask, y1, 0.0), axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = (d2 * inv2) * g3 + b3
    y2 = y2.to(tl.float16).to(tl.float32)

    out = (y2 * scale2).to(tl.float16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


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
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b) * 1.1736
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b) * 1.0344
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, out, self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            x2.stride(0), out.stride(0),
            N, 1.1736, 1.0344, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
