import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 976
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X, W0, W2, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # First RMSNorm
    ms = tl.sum(xf * xf, axis=0) / D
    r = 1.0 / tl.sqrt(ms + EPS)
    y = (xf * r).to(tl.float16)  # cast to fp16 (matches .to(x.dtype))

    w0 = tl.load(W0 + cols, mask=mask, other=0.0)  # fp16
    # fp16 * fp16 in PyTorch uses fp32 opmath then rounds to fp16
    y = (y.to(tl.float32) * w0.to(tl.float32)).to(tl.float16)

    # scalar multiply (fp32 opmath, round to fp16)
    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    # Second RMSNorm
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + EPS)
    z = (yf * r2).to(tl.float16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * stride_y + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x * 1.4509
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x @ self.W3

        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _double_rmsnorm_kernel[(m,)](
            x, self.rms0_w, self.rms2_w, y,
            x.stride(0), y.stride(0),
            D=d, EPS=1e-6, SCALE=1.4509,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y @ self.W3


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
