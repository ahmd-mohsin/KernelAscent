import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 88
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _rmsnorm_gelu_kernel(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)

    # normalize (fp32) -> cast to fp16 -> multiply by fp16 weight -> fp16
    w = tl.load(W + cols, mask=mask, other=0.0)
    xn = (xf * inv).to(tl.float16)
    t = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # exact (erf) GELU computed in fp32, matching PyTorch's opmath for half
    tf = t.to(tl.float32)
    y = 0.5 * tf * (1.0 + tl.math.erf(tf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            return F.gelu(x)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _rmsnorm_gelu_kernel[(m,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
