import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 668
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _softmax_rms_kernel(
    X, W, Y,
    n_cols,
    stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to fp16 (matches torch.softmax output dtype), then upcast for RMS
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)

    ms = tl.sum(pf * pf, axis=0) / n_cols
    r = tl.math.rsqrt(ms + eps)

    y16 = (pf * r).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    y16 = y16 * w  # fp16 multiply, matching (...).to(half) * half_weight

    tl.store(Y + row * stride_y + offs, y16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if (not x.is_cuda) or x.dtype != torch.float16:
            # fallback: reference implementation
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]

        y = torch.empty_like(x2d)
        w = self.rms1_w
        if not w.is_cuda:
            w = w.to(x.device)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _softmax_rms_kernel[(n_rows,)](
            x2d, w, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
