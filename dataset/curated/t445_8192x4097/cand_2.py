import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 445
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _rms_relu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row * N + cols).to(tl.float32)

    # mean of squares in fp32 (matches _xf.pow(2).mean(-1))
    ms = tl.sum(x * x, axis=0) / N
    rs = tl.math.rsqrt(ms + eps)

    # normalize, round to bf16 (matches .to(x.dtype)), then multiply by weight
    xn = (x * rs).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + cols).to(tl.float32)
    y = xn * w

    # relu (applying once is equivalent to applying twice)
    y = tl.maximum(y, 0.0)

    tl.store(Y_ptr + row * N + cols, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmuls stay on cuBLAS tensor cores
        h = x @ self.W0
        h = h @ self.W1
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)
        _rms_relu_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N=N, eps=1e-6, BLOCK=N,
            num_warps=8,
        )
        return out
