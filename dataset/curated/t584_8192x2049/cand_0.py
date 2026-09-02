import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 584
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_gelu_double_rms_kernel(
    x_ptr, w1_ptr, w2_ptr, out_ptr,
    D, stride_row,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # Exact (erf) GELU, computed in fp32, rounded back to bf16 like F.gelu
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # First RMSNorm
    ms1 = tl.sum(g * g, axis=0) / D
    y = (g * tl.math.rsqrt(ms1 + 1e-6)).to(tl.bfloat16)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # Second RMSNorm
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / D
    z = (yf * tl.math.rsqrt(ms2 + 1e-6)).to(tl.bfloat16)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # Final scale
    out = (z.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # Fallback: reference implementation
            x = F.gelu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x * 1.3399

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        _fused_gelu_double_rms_kernel[(n_rows,)](
            x2d, self.rms1_w, self.rms2_w, out,
            d, x2d.stride(0),
            SCALE=1.3399,
            BLOCK=BLOCK,
            num_warps=16,
            num_stages=1,
        )
        return out.view(orig_shape)
