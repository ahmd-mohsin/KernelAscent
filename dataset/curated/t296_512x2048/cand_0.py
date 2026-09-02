import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 296
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_rms_relu_ln_gelu(
    X, W, G, B, Y,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- RMSNorm (computed in fp32, rounded to fp16 like reference) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    xn16 = (x * rrms).to(tl.float16)

    # ---- multiply by rms0_w (fp16 op, single rounding like PyTorch) ----
    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    h16 = xn16 * w  # fp16 result

    # ---- ReLU + scale (scale computed in fp32 opmath, rounded to fp16) ----
    hf = tl.maximum(h16.to(tl.float32), 0.0)
    hf = hf * 1.4834
    h = hf.to(tl.float16).to(tl.float32)  # value entering layer_norm (fp16 rounded)
    h = tl.where(mask, h, 0.0)

    # ---- LayerNorm (fp32 accumulation as PyTorch does for half input) ----
    mean = tl.sum(h, axis=0) / N
    d = tl.where(mask, h - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    yln = d * rstd * g + b
    yln = yln.to(tl.float16).to(tl.float32)  # layer_norm outputs fp16

    # ---- exact GELU: 0.5*x*(1+erf(x/sqrt(2))) ----
    out = 0.5 * yln * (1.0 + tl.math.erf(yln * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            _xf = x.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            h = torch.relu(h) * 1.4834
            h = F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)
            return F.gelu(h)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_rms_relu_ln_gelu[(rows,)](
            x2, self.rms0_w, self.ln3_g, self.ln3_b, y,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
