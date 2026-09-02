import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 654
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_rms_ln_act_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm (compute in fp32, cast to fp16, multiply by fp16 weight in fp16)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w16 = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y16 = (xf * r).to(tl.float16) * w16

    # LayerNorm (fp32 compute, fp16 output)
    yf = y16.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z16 = (d * inv * g + b).to(tl.float16)

    # ReLU (fp16, exact)
    z16 = tl.maximum(z16, 0.0)

    # GELU (erf, fp32 compute, fp16 output)
    zf = z16.to(tl.float32)
    gel = zf * 0.5 * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    g16 = gel.to(tl.float16)

    # scale (fp32 compute, fp16 output)
    out = (g16.to(tl.float32) * 1.3529).to(tl.float16)
    tl.store(Y_ptr + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_ln_act_kernel[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            n, x.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
