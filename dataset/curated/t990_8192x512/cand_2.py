import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 990
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_ln_relu_rms_kernel(
    X, G, B, W, Y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, eps=1e-5)
    mean = tl.sum(x, axis=0) / D_
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D_
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b

    # ReLU (applied twice == once), then round to bf16 as PyTorch would
    y = tl.maximum(y, 0.0)
    y_bf16 = y.to(tl.bfloat16)

    # RMSNorm on the bf16 values cast back to fp32 (matches reference exactly)
    xf = y_bf16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D_
    rrms = 1.0 / tl.sqrt(ms + 1e-6)

    normed = (xf * rrms).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (normed.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y + row * D_ + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_ln_relu_rms_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, self.rms3_w, y,
            d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
