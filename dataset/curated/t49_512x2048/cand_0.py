import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 49
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b0_ptr, w_ptr, out_ptr,
                  D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0)

    # bias add in fp16 (matches torch fp16 add)
    x = x + b0

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * r).to(tl.float16)

    # weight mult + scalar mult in fp16 (matches torch fp16 semantics)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = y * w
    y = y * tl.full((), 1.2274, tl.float16)

    # exact GELU, computed in fp32 (matches torch opmath for half)
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(out_ptr + row * D + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x = x.contiguous().view(-1, d)
        n_rows = x.shape[0]
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(n_rows,)](
            x, self.b0, self.rms1_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
