import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 208
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _scale_relu_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    # scale computed in fp32 then rounded back to bf16 (matches PyTorch opmath),
    # since softmax consumes the bf16 tensor in the reference
    s = (x * scale).to(tl.bfloat16).to(tl.float32)
    # relu (applied twice == once)
    r = tl.maximum(s, 0.0)
    r = tl.where(mask, r, float("-inf"))

    m = tl.max(r, axis=0)
    e = tl.exp(r - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul on tensor cores (same as reference)
        h = torch.matmul(x, self.W0)
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _scale_relu_softmax_kernel[(Mrows,)](
            h, y,
            h.stride(0), y.stride(0),
            N, 1.0675,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
