import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 154
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X, LN_G, LN_B, B3, RMS_W, Y,
    N_COLS, EPS_LN, EPS_RMS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    x16 = tl.load(X + row * N_COLS + offs, mask=mask, other=0.0)
    x = x16.to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # exact GELU (erf), computed in fp32 like ATen, then rounded to fp16
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm in fp32 (ATen accumulate type), output cast to fp16
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N_COLS
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N_COLS
    inv = tl.math.rsqrt(var + EPS_LN)

    g = tl.load(LN_G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + offs, mask=mask, other=0.0).to(tl.float32)
    y16 = (d * inv * g + b).to(tl.float16)

    # bias add in fp16 (matches reference dtype semantics)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0)
    y16 = y16 + b3

    # RMSNorm: fp32 accumulation, rsqrt, cast to fp16, then fp16 weight mul
    yf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N_COLS
    r = tl.math.rsqrt(ms + EPS_RMS)
    w = tl.load(RMS_W + offs, mask=mask, other=0.0)
    out = (yf * r).to(tl.float16) * w

    tl.store(Y + row * N_COLS + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback (reference path)
            x = torch.relu(x)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = x + self.b3
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_kernel[(n_rows,)](
            x2, self.ln2_g, self.ln2_b, self.b3, self.rms4_w, y,
            n_cols, 1e-5, 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
