import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 899
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_rms_softmax_relu(
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

    # RMSNorm in float32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)  # cast back to input dtype

    # scale by weight in fp16 (matches PyTorch half*half)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xn * w).to(tl.float32)

    # softmax with float32 accumulation (matches PyTorch half softmax)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    # relu (no-op after softmax, kept for exactness)
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            return torch.relu(torch.relu(x))

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_rms_softmax_relu[(rows,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
