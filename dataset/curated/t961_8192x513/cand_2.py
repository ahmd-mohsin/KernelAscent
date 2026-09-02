import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 961
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_relu_rms_gelu_kernel(
    X, W, Y,
    D: tl.constexpr,
    stride_xm, stride_ym,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu in input dtype (bf16), matches torch.relu on bf16
    x = tl.maximum(x, 0.0)

    # rmsnorm computed in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = tl.math.rsqrt(ms + eps)
    xn = (xf * inv).to(tl.bfloat16)

    # multiply by weight: torch bf16 mul uses fp32 opmath, then rounds to bf16
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # exact gelu in fp32 (matches torch's bf16 gelu with fp32 opmath)
    yf = y.to(tl.float32)
    g = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y + row * stride_ym + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_relu_rms_gelu_kernel[(m,)](
            x2, self.rms1_w, y,
            d,
            x2.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
