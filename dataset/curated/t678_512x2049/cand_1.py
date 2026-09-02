import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 678
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_bias_scale_rmsnorm(
    X_ptr, B_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (fp16) and bias
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x + b1  (PyTorch half add: compute in fp32, round to fp16)
    v = (x + b).to(tl.float16).to(tl.float32)
    # x * 1.4979 (scalar kept fp32 in opmath, result rounded to fp16)
    v = (v * SCALE).to(tl.float16)

    # RMSNorm in fp32 (matches _xf = x.float())
    vf = v.to(tl.float32)
    sq = tl.where(mask, vf * vf, 0.0)
    ms = tl.sum(sq, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    normed = (vf * r).to(tl.float16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (normed * w).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM with fp32 accumulate (same as reference matmul)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        _fused_bias_scale_rmsnorm[(m,)](
            h, self.b1, self.rms3_w, y,
            n, h.stride(0), y.stride(0),
            SCALE=1.4979,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
