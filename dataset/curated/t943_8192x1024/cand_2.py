import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 943
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_rms_relu_gelu(X, W, Y, N, stride_x, stride_y, eps,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean over N)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    # cast normalized to fp16, multiply by fp16 weight (fp16 arithmetic)
    xn = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float16)
    h = xn * w

    # ReLU in fp16
    zero = tl.zeros_like(h)
    h = tl.maximum(h, zero)

    # exact GELU computed in fp32 (matches PyTorch opmath for half)
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.relu(x)
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_rms_relu_gelu[(Mrows,)](
            x2, self.rms0_w, y, N,
            x2.stride(0), y.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
