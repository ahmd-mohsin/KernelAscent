import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 860
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _ln_softmax3_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, fp16 rounding of output like PyTorch) ----
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    # round to fp16 (layer_norm returns fp16 tensor)
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax x3 (fp32 math, fp16 rounding between ops like PyTorch) ----
    for _ in tl.static_range(3):
        y = tl.where(mask, y, float('-inf'))
        m = tl.max(y, axis=0)
        e = tl.exp(y - m)
        e = tl.where(mask, e, 0.0)
        s = tl.sum(e, axis=0)
        y = e / s
        y = y.to(tl.float16).to(tl.float32)

    tl.store(Y + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4
        if BLOCK >= 1024:
            num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _ln_softmax3_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, y,
            x2.stride(0), y.stride(0),
            D=d, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
