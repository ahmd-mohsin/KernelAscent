import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 20
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, W0, B1, G2, B2, OUT,
    N, stride_x, stride_o,
    eps_rms, eps_ln,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (fp32 math, then cast to fp16 like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    rms = xf * (1.0 / tl.sqrt(ms + eps_rms))
    rms_h = rms.to(tl.float16)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    # fp16 multiply and add (matches reference elementwise ops on fp16)
    y = (rms_h * w0 + b1).to(tl.float16)

    # LayerNorm: stats in fp32, input is fp16 values
    yf = y.to(tl.float32)
    mean = tl.sum(tl.where(mask, yf, 0.0), axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps_ln)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (diff * inv_std) * g2 + b2
    ln_h = ln.to(tl.float16)

    # Softmax in fp32 on fp16 input values
    lf = ln_h.to(tl.float32)
    lf = tl.where(mask, lf, float('-inf'))
    mx = tl.max(lf, axis=0)
    e = tl.exp(lf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.rms0_w, self.b1, self.ln2_g, self.ln2_b, out,
            N, x2.stride(0), out.stride(0),
            1e-6, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
