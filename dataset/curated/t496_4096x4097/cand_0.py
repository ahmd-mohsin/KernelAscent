import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 496
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_scale_rmsnorm_kernel(
    X, W, Y,
    D, stride_x, stride_y,
    eps, s1, s2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    # load row as bf16 -> fp32
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # replicate the two sequential bf16 multiplications exactly:
    # bf16 result after each scale (round-to-nearest-even, same as PyTorch)
    x = (x * s1).to(tl.bfloat16).to(tl.float32)
    x = (x * s2).to(tl.bfloat16).to(tl.float32)

    # RMS statistics in fp32 (matches _xf.pow(2).mean(-1) + rsqrt in fp32)
    ms = tl.sum(x * x, axis=0) / D
    r = tl.rsqrt(ms + eps)

    # normalize in fp32, cast to bf16 (matches .to(x.dtype))
    y = (x * r).to(tl.bfloat16).to(tl.float32)

    # bf16 * bf16 weight (PyTorch computes in fp32 opmath, rounds to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # fallback: reference implementation
            x = x * 1.2952
            x = x * 1.4173
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)

        w = self.rms2_w
        if w.device != x.device:
            w = w.to(x.device)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK <= 8192 else 16

        _fused_scale_rmsnorm_kernel[(m,)](
            x2d, w, y,
            d, x2d.stride(0), y.stride(0),
            1e-6, 1.2952, 1.4173,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
