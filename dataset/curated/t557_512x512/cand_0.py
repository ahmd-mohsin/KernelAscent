import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 557
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, B1, W, Out, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load input row, relu (in fp32 opmath, exact for relu)
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # bias add: fp32 compute, round back to bf16 (matches PyTorch opmath)
    b = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32, cast to bf16, multiply by weight (fp32 compute, bf16 round)
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)

    # relu
    y = tl.maximum(y, 0.0)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.math.exp(y - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(n_rows,)](
            x2, self.b1, self.rms2_w, out,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
