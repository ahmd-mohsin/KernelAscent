import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 675
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_rms_kernel(
    X_ptr, W3_ptr, W4_ptr, Y_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact, erf-based), computed in fp32 then rounded to bf16 (matches PyTorch opmath)
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale by 1.113 (fp32 opmath, round to bf16)
    g = (g * 1.113).to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g2 = g * 0.5 * (1.0 + tl.math.erf(g * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # RMSNorm #1 (fp32 reduction)
    sq = tl.where(mask, g2 * g2, 0.0)
    ms1 = tl.sum(sq, axis=0) / D
    r1 = tl.math.rsqrt(ms1 + 1e-6)
    y = (g2 * r1).to(tl.bfloat16).to(tl.float32)

    w3 = tl.load(W3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w3).to(tl.bfloat16).to(tl.float32)

    # RMSNorm #2 (fp32 reduction)
    sq2 = tl.where(mask, y * y, 0.0)
    ms2 = tl.sum(sq2, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    y2 = (y * r2).to(tl.bfloat16).to(tl.float32)

    w4 = tl.load(W4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y2 * w4).to(tl.bfloat16)

    tl.store(Y_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x * 1.113
            x = F.gelu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_rms_kernel[(n_rows,)](
            x2d, self.rms3_w, self.rms4_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
