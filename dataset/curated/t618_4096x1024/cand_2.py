import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 618
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_scale_relu_softmax(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # x * 1.3747 (compute fp32, round to bf16 to match eager)
    v = x.to(tl.float32) * 1.3747
    v = v.to(tl.bfloat16)
    # x * 1.155
    v = v.to(tl.float32) * 1.155
    v = v.to(tl.bfloat16)
    # relu
    v = tl.maximum(v, 0.0).to(tl.float32)

    # softmax in fp32 (matches PyTorch internal accumulation)
    v = tl.where(mask, v, float('-inf'))
    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        assert x.is_cuda, "expected CUDA tensor"
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _fused_scale_relu_softmax[(m,)](
            x, y,
            x.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
