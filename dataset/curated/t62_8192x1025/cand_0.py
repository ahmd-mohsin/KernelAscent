import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 62
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _relu_softmax_scale_kernel(X, Y, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X + row * stride + cols
    x = tl.load(ptr).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)
    # softmax (fp32 accumulation, matching PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)
    # scale by 1.4325 (opmath in fp32, cast back to fp16, matching eager)
    y = (p.to(tl.float32) * 1.4325).to(tl.float16)
    tl.store(Y + row * stride + cols, y)


@triton.jit
def _rmsnorm_kernel(X, W, Y, stride, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    xf = tl.load(X + row * stride + cols).to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)
    w = tl.load(W + cols)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    tl.store(Y + row * stride + cols, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 via cuBLAS tensor cores
        h = x @ self.W0  # (M, 2048), fp16, contiguous
        m, n = h.shape

        # Fused relu + softmax + scale (in-place on h)
        _relu_softmax_scale_kernel[(m,)](
            h, h, h.stride(0), BLOCK=2048, num_warps=8
        )

        # GEMM 2 via cuBLAS tensor cores
        z = h @ self.W4  # (M, 512), fp16, contiguous

        # Fused RMSNorm (in-place on z)
        _rmsnorm_kernel[(z.shape[0],)](
            z, self.rms5_w, z, z.stride(0), N=512, BLOCK=512, num_warps=4
        )
        return z
