import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 241
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_rms_softmax_ln_kernel(
    X, W, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- load row (fp16 -> fp32) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (compute in fp32, round to fp16, multiply by fp16 weight in fp16) ----
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    xn16 = (x * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)          # fp16
    y16 = xn16 * w                                        # fp16 multiply (matches eager)
    yf = y16.to(tl.float32)

    # ---- softmax (accumulate in fp32, output rounded to fp16 like eager) ----
    yf_masked = tl.where(mask, yf, float('-inf'))
    mx = tl.max(yf_masked, axis=0)
    e = tl.exp(yf_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm16 = (e / s).to(tl.float16)                         # fp16 intermediate (matches eager)
    sm = sm16.to(tl.float32)

    # ---- layer norm (stats in fp32, eps = 1e-5) ----
    mean = tl.sum(tl.where(mask, sm, 0.0), axis=0) / N
    d = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (sm - mean) * rstd * g + b

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_rms_softmax_ln_kernel[(Mrows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, y,
            x2.stride(0), y.stride(0),
            N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
