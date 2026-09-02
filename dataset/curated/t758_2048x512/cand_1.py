import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 758
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, w0_ptr, b1_ptr, w3_ptr, w4_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * r).to(tl.bfloat16).to(tl.float32) * w0
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- add bias ----
    b = tl.load(b1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf) in fp32 opmath, round back to bf16 ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 3 ----
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * r).to(tl.bfloat16).to(tl.float32) * w3
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 4 ----
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w4 = tl.load(w4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * r).to(tl.bfloat16).to(tl.float32) * w4

    tl.store(out_ptr + row * D + offs, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms0_w, self.b1, self.rms3_w, self.rms4_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)

    def _forward_ref(self, x):
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = x + self.b1
        x = F.gelu(x)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        return x
