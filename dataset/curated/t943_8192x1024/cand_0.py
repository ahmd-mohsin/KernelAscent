import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 943
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_rms_relu_gelu(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    # normalize, round to fp16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    # fp16 * fp16 computed in fp32 (PyTorch opmath), rounded to fp16
    xw = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    # relu (exact in fp16)
    xr = tl.maximum(xw, 0.0)
    # gelu: computed in fp32 (PyTorch opmath for half), rounded to fp16
    g = xr.to(tl.float32)
    y = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    tl.store(Y + row * D + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            return F.gelu(torch.relu(x))
        x = x.contiguous()
        orig_shape = x.shape
        Dn = orig_shape[-1]
        x2 = x.view(-1, Dn)
        Mn = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dn)
        _fused_rms_relu_gelu[(Mn,)](x2, self.rms0_w, y, Dn, BLOCK=BLOCK,
                                    num_warps=8 if BLOCK >= 1024 else 4)
        return y.view(orig_shape)
