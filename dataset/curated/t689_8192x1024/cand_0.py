import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 689
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _rms_fused_kernel(
    X, W, Y,
    N,
    stride_x, stride_y,
    eps, scale,
    APPLY_RELU_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # mean of squares in fp32 (matches xf.pow(2).mean(-1))
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    # (xf * rsqrt).to(fp16)
    h16 = (x * r).to(tl.float16)

    # fp16 * fp16 weight -> PyTorch computes in fp32 opmath then rounds to fp16
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    h32 = h16.to(tl.float32) * w

    if APPLY_RELU_SCALE:
        # relu commutes with the fp32->fp16 rounding (monotone, sign preserving)
        h32 = tl.maximum(h32, 0.0)
        h16b = h32.to(tl.float16)
        # fp16 tensor * python-float scalar -> computed in fp32, rounded to fp16
        out = (h16b.to(tl.float32) * scale).to(tl.float16)
    else:
        out = h32.to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.relu(x)
            x = x * 1.0454
            x = x @ self.W3
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        Mrows, N0 = x.shape

        # Fused: RMSNorm + weight + relu + scale
        y = torch.empty_like(x)
        _rms_fused_kernel[(Mrows,)](
            x, self.rms0_w, y,
            N0,
            x.stride(0), y.stride(0),
            1e-6, 1.0454,
            APPLY_RELU_SCALE=True,
            BLOCK=triton.next_power_of_2(N0),
            num_warps=8,
        )

        # fp16 GEMM on tensor cores (same as reference matmul)
        z = y @ self.W3

        # Fused: RMSNorm + weight
        N1 = z.shape[1]
        out = torch.empty_like(z)
        _rms_fused_kernel[(Mrows,)](
            z, self.rms4_w, out,
            N1,
            z.stride(0), out.stride(0),
            1e-6, 1.0,
            APPLY_RELU_SCALE=False,
            BLOCK=triton.next_power_of_2(N1),
            num_warps=4,
        )
        return out
