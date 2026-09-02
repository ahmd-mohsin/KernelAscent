import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 446
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_rms_relu(X, W, Y, n_cols, x_stride, y_stride, eps,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU in fp32, then round to bf16 to match F.gelu output dtype
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32 (matches _xf = x.float(); mean of squares over last dim)
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / n_cols
    r = tl.math.rsqrt(ms + eps)
    n = (g * r).to(tl.bfloat16).to(tl.float32)

    # scale by weight, relu, store as bf16
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = n * w
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * y_stride + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # fallback: reference path
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return torch.relu(x)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_rms_relu[(n_rows,)](
            x2d, self.rms1_w, y,
            n_cols, x2d.stride(0), y.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
