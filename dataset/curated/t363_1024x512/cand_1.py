import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 363
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = x * inv
    # round to bf16, multiply by weight in bf16 (rounded), then gelu in fp32
    xn_bf = xn.to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    v_bf = (xn_bf * w).to(tl.bfloat16)
    v = v_bf.to(tl.float32)
    # exact gelu: 0.5*v*(1+erf(v/sqrt(2)))
    out = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(x)
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, y, N, x.stride(0), y.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
