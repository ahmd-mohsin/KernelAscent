import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 963
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _fused_rms_ln_gelu(
    X, W, G, B, Y,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.3694  (half output, fp32 opmath)
    x = (x * 1.3694).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32, cast to fp16
    ms = tl.sum(x * x, axis=0) / N
    y = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.float16)

    # multiply by rms1_w (half*half with fp32 opmath -> half)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y.to(tl.float32) * w).to(tl.float16).to(tl.float32)
    z = tl.where(mask, z, 0.0)

    # LayerNorm (fp32 internal)
    mean = tl.sum(z, axis=0) / N
    d = tl.where(mask, z - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h = ((z - mean) * rstd * g + b).to(tl.float16).to(tl.float32)

    # GELU (exact erf form, fp32 opmath -> half)
    out = h * 0.5 * (1.0 + tl.math.erf(h * 0.7071067811865476))

    tl.store(Y + row * stride + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_gelu[(Mrows,)](
            x2d, self.rms1_w, self.ln2_g, self.ln2_b, y,
            N, x2d.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
