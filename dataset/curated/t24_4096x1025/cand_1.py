import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 24
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)

    x = tl.load(X_ptr + row * stride_x + offs)

    # x = x * 1.4032  (PyTorch half elementwise uses fp32 opmath, rounds to fp16)
    xf = x.to(tl.float32) * SCALE
    x16 = xf.to(tl.float16)

    # RMSNorm in fp32
    xf2 = x16.to(tl.float32)
    ms = tl.sum(xf2 * xf2, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)
    y16 = (xf2 * r).to(tl.float16)

    # multiply by weight (fp16 op with fp32 opmath)
    w = tl.load(W_ptr + offs)
    out = (y16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # First GEMM via cuBLAS (fastest path on A100)
        h = x @ self.W0  # (M, 512), fp16

        Mrows, N = h.shape
        h = h.contiguous()
        y = torch.empty_like(h)

        _scale_rmsnorm_kernel[(Mrows,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            SCALE=1.4032,
            EPS=1e-6,
            N=N,
            num_warps=4,
        )

        # Second GEMM via cuBLAS
        return y @ self.W3
