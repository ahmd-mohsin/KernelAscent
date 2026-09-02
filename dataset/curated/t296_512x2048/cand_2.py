import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 296
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W, G, B, Y,
                  N, stride_x, stride_y,
                  RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
                  SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (fp32 stats, cast to fp16, multiply by fp16 weight in fp16)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    y16 = (xf * r).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y16 = y16 * w

    # ReLU (fp16)
    y16 = tl.where(y16 > 0, y16, y16 * 0)

    # scale by 1.4834 (opmath fp32, cast back to fp16)
    t16 = (y16.to(tl.float32) * SCALE).to(tl.float16)

    # LayerNorm (fp32 internal)
    tf = t16.to(tl.float32)
    tf = tl.where(mask, tf, 0.0)
    mean = tl.sum(tf, axis=0) / N
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h16 = (diff * rstd * g + b).to(tl.float16)

    # GELU (exact erf, opmath fp32)
    hf = h16.to(tl.float32)
    out = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x2, self.rms0_w, self.ln3_g, self.ln3_b, y,
            N, x2.stride(0), y.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5, SCALE=1.4834,
            BLOCK=BLOCK, num_warps=8,
        )
        return y.view(orig_shape)
