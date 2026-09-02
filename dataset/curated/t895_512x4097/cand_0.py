import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 895
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_ln_rms_gelu_kernel(
    X, G, B, W, Y,
    N, stride_x, stride_y,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * inv_std * g + b
    # round-trip through fp16 (matches reference: layer_norm output is fp16)
    y = y.to(tl.float16).to(tl.float32)

    # RMSNorm on fp32 view of the fp16 values
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)

    t = (y * r).to(tl.float16).to(tl.float32)  # (_xf * rsqrt).to(fp16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    t = (t * w).to(tl.float16).to(tl.float32)  # * rms1_w (fp16 result)

    # exact GELU (erf-based) in fp32, output fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * t * (1.0 + tl.math.erf(t * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_ln_rms_gelu_kernel[(Mrows,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, y,
            N, x2.stride(0), y.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
