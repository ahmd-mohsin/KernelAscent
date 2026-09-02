import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 735
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _bias_scale_softmax_kernel(
    X, B, OUT,
    N, stride_x, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # match PyTorch: add in fp32, round to bf16
    y = (x + b).to(tl.bfloat16).to(tl.float32)
    # multiply in fp32, round to bf16
    z = (y * scale).to(tl.bfloat16).to(tl.float32)

    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _bias_scale_softmax_kernel[(Mrows,)](
            h, self.b1, out,
            N, h.stride(0), out.stride(0),
            1.1961,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
