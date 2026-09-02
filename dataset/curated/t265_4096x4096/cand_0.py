import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 265
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_relu_scale_softmax(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # relu -> * 1.3853 (round to bf16) -> relu (no-op) -> * 1.1094 (round to bf16)
    x = tl.maximum(x, 0.0)
    x = (x.to(tl.float32) * 1.3853).to(tl.bfloat16)
    x = (x.to(tl.float32) * 1.1094).to(tl.bfloat16)

    # softmax in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM in bf16 (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_relu_scale_softmax[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
