import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 585
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_rms_act_kernel(
    x_ptr, w_ptr, out_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load row in fp32 (matches x.float())
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    y = x * r
    # .to(x.dtype) rounding step
    y = y.to(tl.float16).to(tl.float32)

    # * weight (half*half computed in fp32 opmath, rounded to fp16)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.float16).to(tl.float32)

    # * 1.4123 (fp32 opmath, round to fp16)
    y = (y * 1.4123).to(tl.float16).to(tl.float32)

    # relu (exact on fp16 values)
    y = tl.maximum(y, 0.0)

    # * 1.0481 (fp32 opmath, round to fp16)
    y = (y * 1.0481).to(tl.float16).to(tl.float32)

    # exact GELU in fp32 (matches PyTorch half GELU which uses fp32 acc)
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(out_ptr + row * D + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16 and x.shape[-1] == self.rms0_w.numel():
            orig_shape = x.shape
            x2 = x.contiguous().view(-1, orig_shape[-1])
            rows, d = x2.shape
            out = torch.empty_like(x2)
            BLOCK = triton.next_power_of_2(d)
            _fused_rms_act_kernel[(rows,)](
                x2, self.rms0_w, out,
                D=d, BLOCK=BLOCK,
                num_warps=4,
            )
            return out.view(orig_shape)

        # fallback (reference path)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = x * 1.4123
        x = torch.relu(x)
        x = x * 1.0481
        x = F.gelu(x)
        return x
