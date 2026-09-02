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
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (fp16) and bias (fp16)
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b1  (fp16 rounding, computed at fp32 opmath like PyTorch)
    t = (x + b).to(tl.float16)

    # x = x * 1.4979  (fp32 opmath, round to fp16)
    s = (t.to(tl.float32) * 1.4979).to(tl.float16)

    # RMSNorm in fp32
    sf = s.to(tl.float32)
    sq = tl.where(mask, sf * sf, 0.0)
    ms = tl.sum(sq, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)

    # (_xf * rsqrt).to(fp16) * rms3_w  (final mul at fp32 opmath, round to fp16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = (sf * r).to(tl.float16)
    y = (normed.to(tl.float32) * w).to(tl.float16)

    tl.store(Y_ptr + row * stride_row + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM with fp32 accumulate (matches reference matmul)
        h = x @ self.W0
        if not h.is_contiguous():
            h = h.contiguous()

        m, n = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        _fused_bias_scale_rmsnorm[(m,)](
            h, self.b1, self.rms3_w, out,
            n, h.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
