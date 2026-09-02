import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 450
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_softmax_affine_ln(
    X, Y, B1, B3, G, B,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax accumulation), round to fp16
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), 0)
    t = (e / s).to(tl.float16)

    # x = x + b1  (fp16 tensors, computed in fp32 opmath, rounded to fp16)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    t = (t.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)

    # x = x * 1.2334
    t = (t.to(tl.float32) * 1.2334).to(tl.float16)

    # x = x + b3
    b3 = tl.load(B3 + offs, mask=mask, other=0.0)
    t = (t.to(tl.float32) + b3.to(tl.float32)).to(tl.float16)

    # layer_norm in fp32 (matches PyTorch half layer_norm accumulation)
    tf = t.to(tl.float32)
    tf_m = tl.where(mask, tf, 0.0)
    mean = tl.sum(tf_m, 0) / N
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, 0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((tf - mean) * rstd * g + b).to(tl.float16)

    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = torch.softmax(x, dim=-1)
            x = x + self.b1
            x = x * 1.2334
            x = x + self.b3
            return F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_affine_ln[(rows,)](
            x2, y, self.b1, self.b3, self.ln4_g, self.ln4_b,
            N, x2.stride(0), y.stride(0),
            EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
