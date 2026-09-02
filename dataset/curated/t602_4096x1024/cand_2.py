import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 602
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_rms_bias_gelu_kernel(
    X, W, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # RMS norm in fp32 (matches reference: _xf * rsqrt(mean(_xf^2) + eps))
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * r).to(tl.bfloat16)  # round to bf16 like .to(x.dtype)

    w = tl.load(W + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)

    # bf16 elementwise mul (fp32 opmath, round to bf16) — matches PyTorch
    x1 = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    # bf16 add bias
    x2 = (x1.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # exact GELU in fp32, round to bf16
    g = x2.to(tl.float32)
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    x3 = g.to(tl.bfloat16)

    # two scalar multiplies, each rounded to bf16 as in reference
    x4 = (x3.to(tl.float32) * 1.3699).to(tl.bfloat16)
    x5 = (x4.to(tl.float32) * 1.2531).to(tl.bfloat16)

    tl.store(Y + row * D + offs, x5, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x + self.b1
            x = F.gelu(x)
            x = x * 1.3699
            x = x * 1.2531
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_rms_bias_gelu_kernel[(m,)](
            xc, self.rms0_w, self.b1, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
