import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 19
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w1_ptr, w3_ptr, out_ptr, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)  # bf16
    # x * 1.2975 (opmath fp32, round back to bf16)
    x = (x.to(tl.float32) * 1.2975).to(tl.bfloat16)

    # RMSNorm 1
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + EPS)
    x = (xf * inv).to(tl.bfloat16)
    w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0)
    x = (x.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # ReLU
    x = tl.maximum(x, 0.0).to(tl.bfloat16)

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + EPS)
    x = (xf * inv).to(tl.bfloat16)
    w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0)
    x = (x.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * D + cols, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.2975
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = torch.relu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        out = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            xc, self.rms1_w, self.rms3_w, out,
            D=d, EPS=1e-6, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)
