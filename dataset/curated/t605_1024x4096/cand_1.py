import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 605
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_rms_kernel(
    X, W, Out,
    stride_xm, stride_om,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax with float accumulation)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    # round to fp16 (softmax output dtype), then cast back to fp32 for RMS
    y16 = y.to(tl.float16)
    yf = y16.to(tl.float32)

    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)

    normed = (yf * r).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = normed * w
    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_rms_kernel[(m,)](
            h, self.rms2_w, out,
            h.stride(0), out.stride(0),
            n, 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
