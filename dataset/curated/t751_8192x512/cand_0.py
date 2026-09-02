import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 751
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, B1, W2, W3, G4, B4,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulate, round to bf16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    y = (e / s).to(tl.bfloat16)

    # add bias in bf16 semantics (fp32 add of two bf16 is exact, single round)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 1
    yf = tl.where(mask, y.to(tl.float32), 0.0)
    ms = tl.sum(yf * yf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    y = (yf * rstd).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 2
    yf = tl.where(mask, y.to(tl.float32), 0.0)
    ms = tl.sum(yf * yf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    y = (yf * rstd).to(tl.bfloat16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm (fp32 internal, single round to bf16)
    yf = tl.where(mask, y.to(tl.float32), 0.0)
    mean = tl.sum(yf, axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((yf - mean) * inv * g + b).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = torch.softmax(x, dim=-1)
            x = x + self.b1
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, y,
            self.b1, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b,
            x2.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
