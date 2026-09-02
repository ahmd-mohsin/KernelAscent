import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 850
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _relu_scale_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # relu (exact in bf16), then scale in fp32 and round to bf16 (matches
    # PyTorch's bf16 * python-float semantics), then upcast for softmax.
    xf = tl.maximum(xf, 0.0)
    xf = (xf * SCALE).to(tl.bfloat16).to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        if not h.is_cuda:
            h = torch.relu(h) * 1.2179
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _relu_scale_softmax_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            n,
            SCALE=1.2179,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
