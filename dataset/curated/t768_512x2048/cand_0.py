import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 768
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, N, stride_x, stride_y,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, then round to fp16 like torch)
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    s = tl.sum(e, 0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # exact gelu (fp32 opmath, round to fp16)
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # scale (round to fp16)
    g = g * 1.1037
    g = g.to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    gm = tl.where(mask, g, 0.0)
    ms = tl.sum(gm * gm, 0) / N
    r = g * tl.math.rsqrt(ms + 1e-6)
    r = r.to(tl.float16).to(tl.float32)

    # weight multiply + relu
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (r * w).to(tl.float16)
    y = tl.maximum(y, y * 0)

    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.rms3_w, y, N,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
