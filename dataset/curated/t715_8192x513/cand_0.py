import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 715
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    # match reference: normalize in fp32, cast to fp16, multiply by fp16 weight (fp16 arithmetic)
    xn = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float16)
    h = xn * w  # fp16 multiply, matches reference dtype semantics
    # gelu computed in fp32 (matches PyTorch CUDA opmath for half)
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    tl.store(Y + row * stride + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _rms_gelu_kernel[(m,)](
            x, self.rms1_w, y, n, x.stride(0), 1e-6,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
