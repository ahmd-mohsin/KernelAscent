import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 653
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w0_ptr, w1_ptr, b4_ptr, out_ptr,
                  D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(b4_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 0
    ms = tl.sum(x * x, axis=0) / D
    x = x * tl.math.rsqrt(ms + 1e-6)
    x = x.to(tl.bfloat16).to(tl.float32)          # .to(x.dtype)
    x = (x * w0).to(tl.bfloat16).to(tl.float32)   # * rms0_w (bf16 result)

    # RMSNorm 1
    ms = tl.sum(x * x, axis=0) / D
    x = x * tl.math.rsqrt(ms + 1e-6)
    x = x.to(tl.bfloat16).to(tl.float32)
    x = (x * w1).to(tl.bfloat16).to(tl.float32)

    # GELU (exact, erf-based) twice, rounding to bf16 after each like PyTorch
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # bias add
    x = (x + b4).to(tl.bfloat16)
    tl.store(out_ptr + row * D + offs, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.gelu(x)
            x = F.gelu(x)
            return x + self.b4

        x = x.contiguous()
        Mrows, Dcols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms0_w, self.rms1_w, self.b4, out,
            D=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
