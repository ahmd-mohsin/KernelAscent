import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 763
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _ln_rms_kernel(
    X, G, B, W, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch fp16 layer_norm)
    mean = tl.sum(x, axis=0) / D_
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    invstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * invstd * g + b
    y16 = y.to(tl.float16)

    # RMSNorm on the fp16-rounded layernorm output (fp32 math)
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)

    z16 = (yf * r).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = z16 * w  # fp16 multiply, matching reference dtype semantics

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _ln_rms_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, y,
            x2.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
