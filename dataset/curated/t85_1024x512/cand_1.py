import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 85
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _ln_rms_scale_kernel(
    X, G, B, W, Y,
    N, stride,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch's mixed-precision layer_norm)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y16 = (d * rstd * g + b).to(tl.float16)  # round to fp16 like layer_norm output

    # RMSNorm: computed on the fp16-rounded layernorm output, in fp32
    yf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps_rms)
    z16 = (yf * r).to(tl.float16)  # matches .to(x.dtype) rounding

    # * rms2_w in half-op (fp32 opmath, fp16 rounding), then * 1.3885 same way
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    t16 = (z16.to(tl.float32) * w).to(tl.float16)
    out = (t16.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_rms_scale_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, y,
            N, h.stride(0),
            1e-5, 1e-6, 1.3885,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
