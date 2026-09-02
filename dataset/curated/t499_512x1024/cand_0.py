import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 499
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_bias_relu_double_rms_kernel(
    x_ptr, b0_ptr, w2_ptr, w3_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0)

    # x = relu(x + b0)  (add in fp32, round to bf16 to match torch bf16 add)
    t = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    t = tl.maximum(t, 0.0).to(tl.bfloat16)

    # RMSNorm 1
    xf = t.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    n1 = (xf * inv).to(tl.bfloat16)
    # bf16 * bf16 computed in fp32 then rounded (matches torch opmath)
    y1 = (n1.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 2
    yf = y1.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / D
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    n2 = (yf * inv2).to(tl.bfloat16)
    y2 = (n2.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        orig_shape = x.shape
        Dm = orig_shape[-1]
        x2d = x.view(-1, Dm)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(Dm)
        _fused_bias_relu_double_rms_kernel[(rows,)](
            x2d, self.b0, self.rms2_w, self.rms3_w, out,
            D=Dm, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
