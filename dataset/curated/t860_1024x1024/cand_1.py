import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 860
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_ln_softmax3_kernel(
    X, Y, G, B,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch on fp16 input)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # round to fp16 (layer_norm output dtype), back to fp32 for softmax
    y = y.to(tl.float16).to(tl.float32)

    # Softmax 1
    y = tl.where(mask, y, float('-inf'))
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    y = e1 / tl.sum(e1, axis=0)
    y = y.to(tl.float16).to(tl.float32)

    # Softmax 2
    yv = tl.where(mask, y, float('-inf'))
    m2 = tl.max(yv, axis=0)
    e2 = tl.exp(yv - m2)
    e2 = tl.where(mask, e2, 0.0)
    y = e2 / tl.sum(e2, axis=0)
    y = y.to(tl.float16).to(tl.float32)

    # Softmax 3
    yv = tl.where(mask, y, float('-inf'))
    m3 = tl.max(yv, axis=0)
    e3 = tl.exp(yv - m3)
    e3 = tl.where(mask, e3, 0.0)
    y = e3 / tl.sum(e3, axis=0)

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


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
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_softmax3_kernel[(rows,)](
            x2, out, self.ln0_g, self.ln0_b,
            x2.stride(0), out.stride(0),
            N=n, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
