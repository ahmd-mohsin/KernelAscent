import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 528
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, B3, W, Out,
    D_size,
    stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_size

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ----- LayerNorm (fp32 math, matching PyTorch upcast behavior) -----
    n = D_size.to(tl.float32)
    mean = tl.sum(x, axis=0) / n
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    # round to bf16 (matches intermediate storage in reference)
    y = y.to(tl.bfloat16).to(tl.float32)

    # ----- scale -----
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # ----- relu -----
    y = tl.maximum(y, 0.0)

    # ----- add bias -----
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)

    # ----- RMSNorm (fp32 reduction like reference) -----
    y_masked = tl.where(mask, y, 0.0)
    ms = tl.sum(y_masked * y_masked, axis=0) / n
    r = 1.0 / tl.sqrt(ms + 1e-6)

    z = (y * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.bfloat16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, self.b3, self.rms4_w, out,
            d,
            x2.stride(0), out.stride(0),
            SCALE=1.3076,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
