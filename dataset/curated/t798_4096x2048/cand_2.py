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
    D: tl.constexpr,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch's mixed-precision layer_norm)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = tl.rsqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    # round to bf16 exactly like the layer_norm output tensor
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 on the bf16-rounded values (matches reference _xf path)
    yz = tl.where(mask, y, 0.0)
    ms = tl.sum(yz * yz, axis=0) / D
    r = tl.rsqrt(ms + eps_rms)
    y = (y * r).to(tl.bfloat16).to(tl.float32)  # .to(x.dtype) rounding

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)  # bf16 elementwise mul rounding
    y = (y * scale).to(tl.bfloat16)             # bf16 scalar mul rounding

    tl.store(Y + row * D + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_ln_rms_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, y,
            d, 1e-5, 1e-6, 1.4469,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
