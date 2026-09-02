import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 259
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_bias_rms_gelu(X, B, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)

    # bias add: fp32 math, round to bf16 (matches PyTorch bf16 add)
    s = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = s.to(tl.float32)
    mean = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(mean + 1e-6)
    n = (xf * inv).to(tl.bfloat16)

    # multiply by weight: fp32 math, round to bf16 (matches PyTorch bf16 mul)
    v = (n.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # exact GELU (erf) computed in fp32 (matches PyTorch opmath for bf16)
    vf = v.to(tl.float32)
    g = 0.5 * vf * (1.0 + tl.math.erf(vf * 0.7071067811865476))

    tl.store(Y + row * D + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(x)
        x = x.contiguous()
        rows = x.numel() // x.shape[-1]
        D_ = x.shape[-1]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D_)
        _fused_bias_rms_gelu[(rows,)](
            x, self.b0, self.rms1_w, y, D_, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
