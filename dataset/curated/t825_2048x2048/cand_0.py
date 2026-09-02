import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 825
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, B0, G1, Bt1, W2, Y,
    stride_xm, stride_ym,
    N, LN_EPS, RMS_EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # x = x + b0 in fp16 (round to fp16 to match reference)
    xh = (x + b0).to(tl.float16)
    xf = xh.to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch native for fp16)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + LN_EPS)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    bt1 = tl.load(Bt1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * inv * g1 + bt1
    yh = y.to(tl.float16)  # layer_norm output cast to fp16

    # RMSNorm: cast fp16 -> fp32, normalize, cast back to fp16, then * w in fp16
    yf = yh.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    zh = (yf * r).to(tl.float16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    out = zh * w2  # fp16 multiply

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, y,
            x.stride(0), y.stride(0),
            n, 1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
