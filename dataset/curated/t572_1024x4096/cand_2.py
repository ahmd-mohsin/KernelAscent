import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 572
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B0, W2, G3, B3, OUT,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0)

    # x = relu(x + b0)  (bf16 add semantics: fp32 compute, round to bf16)
    x = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    x = tl.maximum(x, 0.0).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * inv).to(tl.bfloat16)

    # multiply by rms2_w (bf16 mul: fp32 compute, round back to bf16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm (fp32 internal, bf16 output)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / D_
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (yf - mean) * rstd * g + b

    tl.store(OUT + row * D_ + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x, self.b0, self.rms2_w, self.ln3_g, self.ln3_b, out,
            d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
