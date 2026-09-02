import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 694
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, OUT,
    W0, B1, G2, BT2, B4,
    N, stride_x, stride_o,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32, output rounded to fp16 then scaled by w0 (fp16 mul), + b1 (fp16 add)
    ms = tl.sum(x * x, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(ms + RMS_EPS)
    y = (x * rstd0).to(tl.float16)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    y = (y * w0).to(tl.float16)
    y = (y + b1).to(tl.float16)

    # LayerNorm in fp32
    yf = y.to(tl.float32)
    mean = tl.sum(tl.where(mask, yf, 0.0), axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    bt2 = tl.load(BT2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((yf - mean) * rstd * g2 + bt2).to(tl.float16)

    # ReLU + b4 (fp16 add)
    z = tl.maximum(z, 0.0)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    out = (z + b4).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = y + self.b1
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = torch.relu(y)
            return y + self.b4

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(rows,)](
            x2, out,
            self.rms0_w, self.b1, self.ln2_g, self.ln2_b, self.b4,
            N, x2.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
