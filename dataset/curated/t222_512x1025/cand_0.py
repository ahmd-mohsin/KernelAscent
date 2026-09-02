import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 222
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _gelu_rmsnorm_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact (erf-based) GELU computed in fp32 (matches PyTorch opmath), rounded to fp16
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # RMSNorm over the fp16 gelu output, promoted to fp32
    gf = g16.to(tl.float32)
    ms = tl.sum(gf * gf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    normed16 = (gf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (normed16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (fastest path for the matmul)
        h = x @ self.W0

        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK_N = triton.next_power_of_2(N)
        _gelu_rmsnorm_kernel[(Mrows,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
