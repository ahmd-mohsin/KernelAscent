import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 291
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _rms_softmax_kernel(
    x_ptr, w_ptr, out_ptr,
    D: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = x * inv
    # cast to bf16 (matches .to(x.dtype))
    xn_bf = xn.to(tl.bfloat16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # elementwise mul computed in fp32, rounded to bf16 (matches PyTorch bf16 mul)
    y = (xn_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax with fp32 accumulation (matches PyTorch bf16 softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(out_ptr + row * D + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _rms_softmax_kernel[(m,)](
            x2, self.rms0_w, out,
            d, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
