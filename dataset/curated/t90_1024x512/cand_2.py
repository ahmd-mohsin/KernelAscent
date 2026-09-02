import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 90
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_rms_gelu_kernel(
    X, W, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # Load row in fp32 (matches x.float())
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # cast normalized value back to bf16 (matches .to(x.dtype))
    xn = (x * inv).to(tl.bfloat16).to(tl.float32)

    # multiply by weight: bf16 op computed in fp32, rounded to bf16
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    t = (xn * w).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), computed in fp32, rounded to bf16
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # add bias, round to bf16
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    s = (g + b).to(tl.bfloat16).to(tl.float32)

    # scale, round to bf16
    out = (s * 1.0205).to(tl.bfloat16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = F.gelu(x)
            x = x + self.b2
            x = x * 1.0205
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_rms_gelu_kernel[(rows,)](
            x2, self.rms0_w, self.b2, y,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
