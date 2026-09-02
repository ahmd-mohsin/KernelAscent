import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 674
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_rms_ln_relu(
    X, RMSW, G, B, Y,
    N, stride_x, stride_y,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32, round to fp16, multiply by fp16 weight (fp16 mul)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + RMS_EPS)
    xh = (xf * inv).to(tl.float16)

    w = tl.load(RMSW + cols, mask=mask, other=0.0).to(tl.float16)
    xh = xh * w  # fp16 multiply, matches PyTorch behavior

    # LayerNorm: computed in fp32 (matches PyTorch half layer_norm)
    xf2 = xh.to(tl.float32)
    mean = tl.sum(xf2, axis=0) / N
    diff = tl.where(mask, xf2 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf2 - mean) * rstd * g + b

    # ReLU
    y = tl.maximum(y, 0.0).to(tl.float16)
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        x = x.contiguous()
        y = torch.empty_like(x)
        _fused_rms_ln_relu[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            n, x.stride(0), y.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=512, num_warps=4,
        )
        return y
