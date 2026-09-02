import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 162
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_relu_rms_ln_kernel(
    X, B1, W2, G3, B3, B4, Y,
    D: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    base = row * D

    x = tl.load(X + base + offs).to(tl.float32)
    b1 = tl.load(B1 + offs).to(tl.float32)

    # relu + bias (bf16 rounding as in reference elementwise ops)
    x = tl.maximum(x, 0.0) + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # RMSNorm (fp32 math, cast to bf16, then bf16-rounded scale)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + EPS_RMS)
    x = (x * inv).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + offs).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation, bf16 output like PyTorch native kernel)
    mean = tl.sum(x, axis=0) / D
    xm = x - mean
    var = tl.sum(xm * xm, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G3 + offs).to(tl.float32)
    b3 = tl.load(B3 + offs).to(tl.float32)
    y = xm * rstd * g + b3
    y = y.to(tl.bfloat16).to(tl.float32)

    # final bias add (bf16 rounding)
    b4 = tl.load(B4 + offs).to(tl.float32)
    y = y + b4
    tl.store(Y + base + offs, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = torch.relu(x)
            x = x + self.b1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = x + self.b4
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        y = torch.empty_like(xc)

        _fused_relu_rms_ln_kernel[(n_rows,)](
            xc, self.b1, self.rms2_w, self.ln3_g, self.ln3_b, self.b4, y,
            D=d, EPS_RMS=1e-6, EPS_LN=1e-5,
            num_warps=8,
        )
        return y.view(orig_shape)
