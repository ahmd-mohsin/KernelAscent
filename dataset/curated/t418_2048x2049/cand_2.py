import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 418
M, D, DT = 2048, 2049, torch.float16


@triton.jit
def _fused_ln_softmax2_kernel(
    X_ptr, B0_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b0 = tl.load(B0_ptr + offs, mask=mask, other=0.0)

    # x = x + b0 in fp16 (match reference rounding), then compute LN in fp32
    xh = (x + b0)  # fp16 add
    xf = xh.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)

    n = N.to(tl.float32)
    mean = tl.sum(xf, axis=0) / n
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b

    # round to fp16 (layer_norm output dtype), then softmax #1 in fp32
    y = y.to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1

    # round to fp16 (softmax output dtype), then softmax #2 in fp32
    p1 = p1.to(tl.float16).to(tl.float32)
    p1 = tl.where(mask, p1, float('-inf'))
    m2 = tl.max(p1, axis=0)
    e2 = tl.exp(p1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    tl.store(Y_ptr + row * stride_y + offs, p2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.softmax(y, dim=-1)
            y = torch.softmax(y, dim=-1)
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_ln_softmax2_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, out,
            N, x2.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
