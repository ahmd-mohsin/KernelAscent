import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 997
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_ln_softmax_kernel(
    X, Y, G, B, B1, B3,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm statistics (fp32 accumulation, like ATen)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)

    # LN output rounded to fp16 (reference stores fp16 intermediate)
    t = (x - mean) * rstd * g + b
    t = t.to(tl.float16).to(tl.float32)

    # add b1 in fp16 precision
    t = (t + b1)
    t = t.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation
    t = tl.where(mask, t, float("-inf"))
    mx = tl.max(t, axis=0)
    e = tl.exp(t - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    sm = sm.to(tl.float16).to(tl.float32)

    out = (sm + b3).to(tl.float16)
    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax_kernel[(rows,)](
            x2, y, self.ln0_g, self.ln0_b, self.b1, self.b3,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
