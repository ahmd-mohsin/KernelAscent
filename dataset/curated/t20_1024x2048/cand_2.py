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
    rms_eps, ln_eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptr = X + row * stride_x + cols
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    # --- RMSNorm (fp32) ---
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + rms_eps)
    xn = (x * inv).to(tl.float16)

    # --- scale by rms0_w and add b1 (fp16 arithmetic, matching PyTorch) ---
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    y = xn * w0 + b1  # fp16

    # --- LayerNorm (fp32 internal) ---
    yf = y.to(tl.float32)
    mean = tl.sum(tl.where(mask, yf, 0.0), axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + ln_eps)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (diff * rstd) * g2 + b2
    z = z.to(tl.float16)  # layer_norm output rounded to fp16

    # --- Softmax (fp32 internal) ---
    zf = tl.where(mask, z.to(tl.float32), float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
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
