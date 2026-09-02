import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 621
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_ln_relu_ln_rms_gelu(
    X, Y, G0, B0, G2, B2, W3,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, bf16 output like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xc * rstd) * g0 + b0).to(tl.bfloat16)

    # ---- ReLU ----
    y = tl.maximum(y, y - y)  # max(y, 0) in bf16

    # ---- LayerNorm 2 ----
    x = y.to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xc * rstd) * g2 + b2).to(tl.bfloat16)

    # ---- RMSNorm (fp32 accumulate, bf16 cast, then bf16 weight multiply) ----
    xf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y = (xf * r).to(tl.bfloat16)
    w = tl.load(W3 + cols, mask=mask, other=0.0)
    y = y * w  # bf16 elementwise multiply (matches PyTorch mul on bf16)

    # ---- GELU (erf-based, fp32 internal math like PyTorch CUDA kernel) ----
    t = y.to(tl.float32)
    out = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = F.gelu(x)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_relu_ln_rms_gelu[(rows,)](
            x2d, y,
            self.ln0_g, self.ln0_b,
            self.ln2_g, self.ln2_b,
            self.rms3_w,
            N, x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
