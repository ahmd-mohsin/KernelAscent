import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 812
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_triple_rmsnorm(
    x_ptr, w0_ptr, w2_ptr, w3_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms = tl.sum(x * x, axis=0) / D
    y = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    y = (y * w0).to(tl.bfloat16).to(tl.float32)

    # ---- scale ----
    y = (y * 1.4041).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms = tl.sum(y * y, axis=0) / D
    y = (y * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    y = (y * w2).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 3 ----
    ms = tl.sum(y * y, axis=0) / D
    y = (y * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    y = (y * w3).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x * 1.4041
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_triple_rmsnorm[(n_rows,)](
            x2, self.rms0_w, self.rms2_w, self.rms3_w, out,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
