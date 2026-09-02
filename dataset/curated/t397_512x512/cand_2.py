import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 397
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(X, W_rms, G, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_h = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    x = x_h.to(tl.float32)

    # RMSNorm (fp32)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn_h = (x * inv).to(tl.float16)

    # multiply by weight in fp16 (matches reference fp16 arithmetic)
    w = tl.load(W_rms + cols, mask=mask, other=0.0).to(tl.float16)
    xw = xn_h * w

    # ReLU in fp16
    zero = tl.zeros_like(xw)
    xr = tl.where(xw > zero, xw, zero)

    # LayerNorm in fp32
    xf = tl.where(mask, xr.to(tl.float32), 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (xf - mean) * rstd * g + b
    ln_h = ln.to(tl.float16)

    # GELU (erf, fp32 opmath) on fp16-rounded LN output
    v = ln_h.to(tl.float32)
    out = v * 0.5 * (1.0 + tl.math.erf(v * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, y,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return y.view(orig_shape)
