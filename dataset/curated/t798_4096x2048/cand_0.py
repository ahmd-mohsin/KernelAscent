import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 798
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Y,
    N, stride_x, stride_y,
    ln_eps, rms_eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch for bf16 inputs)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    inv = tl.math.rsqrt(var + ln_eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = xc * inv * g + b
    # round to bf16 (layer_norm output dtype), then upcast as the reference does
    y1 = y1.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, y1 * y1, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + rms_eps)
    z = (y1 * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.bfloat16).to(tl.float32)
    out = (out * scale).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_rms_kernel[(Mrows,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, y,
            N, x2.stride(0), y.stride(0),
            1e-5, 1e-6, 1.4469,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
