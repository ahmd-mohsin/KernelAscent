import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 982
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_kernel(
    X, OUT, W1, W2, B3, G4, B4,
    stride_xm, stride_om,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu (fp16)
    x = tl.maximum(x, 0.0)

    # rmsnorm 1
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    x = xh * w1  # fp16 multiply

    # rmsnorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    x = xh * w2

    # bias add (fp16)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    x = x + b3

    # layernorm (fp32 internal, like PyTorch for half)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b

    tl.store(OUT + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            xx = torch.relu(x)
            _xf = xx.float(); xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms1_w
            _xf = xx.float(); xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms2_w
            xx = xx + self.b3
            return F.layer_norm(xx, (xx.shape[-1],), self.ln4_g, self.ln4_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x2, out,
            self.rms1_w, self.rms2_w, self.b3, self.ln4_g, self.ln4_b,
            x2.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)
