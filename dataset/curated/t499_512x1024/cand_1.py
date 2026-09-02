import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 499
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b0_ptr, w2_ptr, w3_ptr, out_ptr,
                  D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x = relu(x + b0), rounded to bf16 like the eager op
    x = (x + b0).to(tl.bfloat16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # RMSNorm 1
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    t = (x * inv).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = (t * w2).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ms2 = tl.sum(t * t, axis=0) / D
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    u = (t * inv2).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (u * w3).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows = x.numel() // x.shape[-1]
        d = x.shape[-1]
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x, self.b0, self.rms2_w, self.rms3_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
