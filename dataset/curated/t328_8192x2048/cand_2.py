import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 328
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_rms_softmax2_rms_relu(
    X, W0, W3, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # ---- RMSNorm 0 ----
    ms = tl.sum(xf * xf, axis=0) / D
    rstd = tl.math.rsqrt(ms + 1e-6)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    y16 = (xf * rstd).to(tl.float16) * w0  # fp16 multiply, matches PyTorch

    # ---- Softmax 1 (fp32 accumulation, fp16 output; matches PyTorch CUDA path) ----
    yf = tl.where(mask, y16.to(tl.float32), float('-inf'))
    mx = tl.max(yf, axis=0)
    e = tl.exp(yf - mx)
    s = tl.sum(e, axis=0)
    y16 = (e / s).to(tl.float16)

    # ---- Softmax 2 ----
    yf = tl.where(mask, y16.to(tl.float32), float('-inf'))
    mx = tl.max(yf, axis=0)
    e = tl.exp(yf - mx)
    s = tl.sum(e, axis=0)
    y16 = (e / s).to(tl.float16)

    # ---- RMSNorm 3 ----
    yf = tl.where(mask, y16.to(tl.float32), 0.0)
    ms = tl.sum(yf * yf, axis=0) / D
    rstd = tl.math.rsqrt(ms + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    out = (yf * rstd).to(tl.float16) * w3

    # ---- ReLU ----
    out = tl.maximum(out, 0.0).to(tl.float16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        d = x.shape[-1]
        rows = x.numel() // d
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_rms_softmax2_rms_relu[(rows,)](
            x, self.rms0_w, self.rms3_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
