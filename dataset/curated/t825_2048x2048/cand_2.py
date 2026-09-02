import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 825
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, B0, G1, B1, W2, Y,
    N, stride_x, stride_y,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # bias add in fp16 (matches x + self.b0 in half)
    t = (x + b0).to(tl.float16)
    tf = t.to(tl.float32)

    # LayerNorm (fp32 internal, like PyTorch on half input)
    mean = tl.sum(tf, axis=0) / N
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + LN_EPS)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    ln_out = ((diff * inv_std) * g1 + b1).to(tl.float16)

    # RMSNorm: upcast to fp32, normalize, cast to fp16, then multiply by w in fp16
    xf = ln_out.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    normed = (xf * r).to(tl.float16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    out = normed * w2

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, y,
            N, x2.stride(0), y.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
