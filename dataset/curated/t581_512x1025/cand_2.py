import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 581
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # round to bf16 as torch.softmax would output bf16
    sm_bf = sm.to(tl.bfloat16)
    xf = sm_bf.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + eps)
    normed = (xf * rs).to(tl.bfloat16)

    # multiply by weight (fp32 opmath, bf16 output — matches CUDA bf16 mul)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (normed.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xs = torch.softmax(x, dim=-1)
            _xf = xs.float()
            return (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xs.dtype) * self.rms1_w

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _softmax_rms_kernel[(Mrows,)](
            x2, self.rms1_w, y,
            x2.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
