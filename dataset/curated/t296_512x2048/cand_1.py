import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 296
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, g_ptr, b_ptr, out_ptr,
    N, stride,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    x = x16.to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16 like reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + RMS_EPS)
    y = (x * inv).to(tl.float16).to(tl.float32)

    # * rms0_w  (fp16 tensor mul -> opmath fp32, result rounded to fp16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.float16).to(tl.float32)

    # relu
    y = tl.maximum(y, 0.0)

    # * 1.4834  (opmath fp32, rounded to fp16)
    y = (y * 1.4834).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 internal math, fp16 output)
    yv = tl.where(mask, y, 0.0)
    mean = tl.sum(yv, axis=0) / N
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((y - mean) * rstd * g + b).to(tl.float16).to(tl.float32)

    # GELU (exact, erf; fp32 math, fp16 output)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = z * 0.5 * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(out_ptr + row * stride + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = torch.relu(y) * 1.4834
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return F.gelu(y)

        x = x.contiguous()
        shape = x.shape
        N = shape[-1]
        x2d = x.view(-1, N)
        Mrows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x2d, self.rms0_w, self.ln3_g, self.ln3_b, out,
            N, x2d.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(shape)
