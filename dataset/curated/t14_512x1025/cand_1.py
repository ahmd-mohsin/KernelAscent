import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 14
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_relu_rms_rms_relu(
    x_ptr, w1_ptr, w2_ptr, out_ptr,
    D, stride_row,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load row, upcast to fp32, relu
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # --- RMSNorm 1 ---
    ms1 = tl.sum(x * x, axis=0) / D
    r1 = tl.math.rsqrt(ms1 + EPS)
    y = (x * r1).to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((y * w1).to(tl.bfloat16)).to(tl.float32)

    # --- RMSNorm 2 ---
    ms2 = tl.sum(y * y, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + EPS)
    z = (y * r2).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z * w2).to(tl.bfloat16)

    # final relu
    z = tl.maximum(z, 0.0)
    tl.store(out_ptr + row * stride_row + offs, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: original computation
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return torch.relu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_rms_rms_relu[(n_rows,)](
            x2d, self.rms1_w, self.rms2_w, out,
            d, x2d.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
