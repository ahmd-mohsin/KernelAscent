import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 662
M, D, DT = 512, 513, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w0_ptr, b1_ptr, w2_ptr, out_ptr,
                  n_cols, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm 0
    ms = tl.sum(xf * xf, axis=0) / n_cols
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.bfloat16)  # cast to bf16 like reference

    w0 = tl.load(w0_ptr + cols, mask=mask, other=0.0)
    # bf16 * bf16 in PyTorch computes in fp32 then rounds back to bf16
    y = (xn.to(tl.float32) * w0.to(tl.float32)).to(tl.bfloat16)

    # add bias (bf16 add via fp32 opmath, rounded to bf16)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 2
    yf = y.to(tl.float32)
    yf_masked = tl.where(mask, yf, 0.0)
    ms2 = tl.sum(yf_masked * yf_masked, axis=0) / n_cols
    inv2 = tl.math.rsqrt(ms2 + 1e-6)
    yn = (yf * inv2).to(tl.bfloat16)

    w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0)
    z = (yn.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # exact GELU (erf), computed in fp32 (opmath), rounded to bf16
    zf = z.to(tl.float32)
    g = zf * 0.5 * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x + self.b1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return F.gelu(x)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(n_rows,)](
            x2, self.rms0_w, self.b1, self.rms2_w, out,
            n_cols, x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return out.view(orig_shape)
