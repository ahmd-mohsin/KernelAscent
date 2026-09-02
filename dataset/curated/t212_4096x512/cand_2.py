import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 212
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_kernel(
    X, B0, G, B, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)                 # fp16

    # add + scale in fp16 (match reference dtype semantics)
    x = x + b0
    scale: tl.constexpr = 1.4934
    x = (x.to(tl.float32) * scale).to(tl.float16)

    # LayerNorm in fp32 (matches PyTorch fp16 layer_norm internals)
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + bb
    y16 = y.to(tl.float16)  # layer_norm outputs fp16

    # RMSNorm: cast to fp32, normalize, cast to fp16, mul by weight in fp16
    yf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    z16 = (yf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    out = z16 * w

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x2, self.b0, self.ln2_g, self.ln2_b, self.rms3_w, y,
            x2.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
