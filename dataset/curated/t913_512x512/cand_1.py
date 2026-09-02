import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 913
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_scale_gelu_rms_kernel(
    x_ptr, w_ptr, out_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.3419 (bf16 rounding), then x = x * 1.3693 (bf16 rounding)
    x = (x * 1.3419).to(tl.bfloat16).to(tl.float32)
    x = (x * 1.3693).to(tl.bfloat16).to(tl.float32)

    # exact GELU computed in fp32 (matches PyTorch opmath), rounded to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    gm = tl.where(mask, g, 0.0)
    ms = tl.sum(gm * gm, axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)

    y = (g * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.3419
            x = x * 1.3693
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        out = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        _fused_scale_gelu_rms_kernel[(n_rows,)](
            xc, self.rms3_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)
