import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 138
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    # scale in fp32 then round to bf16 (matches PyTorch bf16 mul semantics)
    xf = x.to(tl.float32) * 1.0731
    xb = xf.to(tl.bfloat16)
    # relu
    xb = tl.maximum(xb, 0.0)
    # rmsnorm in fp32
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    yn = (xf * rstd).to(tl.bfloat16)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    out = (yn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.rms2_w, out, d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
