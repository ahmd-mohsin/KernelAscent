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
    stride_x, stride_y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch bf16 layer_norm)
    mean = tl.sum(x, axis=0) / D_
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D_
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b

    # cast to bf16 (layer_norm output dtype), then relu
    y = y.to(tl.bfloat16)
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)

    # RMSNorm: compute in fp32 on the bf16 values
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / D_
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    yn = (yf * rrms).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = yn * w

    tl.store(Y + row * stride_y + cols, out, mask=mask)


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
            x2.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
