import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 147
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_gelu2_rms_relu(
    X, W, Y,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- load row (upcast to fp32, matching PyTorch opmath) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, cast result back to bf16) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- gelu (exact, erf) x2, each rounds back through bf16 ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.bfloat16).to(tl.float32)
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32 math, cast to bf16, then bf16 * bf16 weight) ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    yb = (y * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    yb = (yb.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # ---- relu ----
    zero = tl.zeros_like(yb)
    out = tl.where(yb > zero, yb, zero)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference fallback
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return torch.relu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_softmax_gelu2_rms_relu[(Mrows,)](
            x2, self.rms3_w, y,
            N,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
