import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 797
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    G1, B1, W2, G3, B3, W4,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # ReLU
    x = tl.maximum(x, 0.0)

    n = N.to(tl.float32)

    # ---- LayerNorm 1 (eps=1e-5), fp32 math, bf16 rounding of output ----
    mean = tl.sum(x, axis=0) / n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd) * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 2 (eps=1e-6) ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / n
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (y * r).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w2).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 (eps=1e-5) ----
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / n
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd) * g3 + b3
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 4 (eps=1e-6) ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / n
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (y * r).to(tl.bfloat16).to(tl.float32)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w4).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # Fallback: reference implementation
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.reshape(-1, N)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 16 if BLOCK >= 8192 else 8

        _fused_norm_kernel[(Mrows,)](
            x2d, y,
            self.ln1_g, self.ln1_b, self.rms2_w,
            self.ln3_g, self.ln3_b, self.rms4_w,
            N, x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
