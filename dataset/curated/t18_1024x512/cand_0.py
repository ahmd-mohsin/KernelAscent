import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 18
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_scale_relu_softmax(
    X, Y,
    N,
    stride_xm, stride_ym,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # x * 1.2399 with bf16 rounding (match eager bf16 elementwise op)
    v = (x.to(tl.float32) * S1).to(tl.bfloat16)
    # relu
    v = tl.maximum(v, 0.0)
    # * 1.1622 with bf16 rounding
    v = (v.to(tl.float32) * S2).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch's internal fp32 accumulation)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mr, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_scale_relu_softmax[(Mr,)](
            h, y, N,
            h.stride(0), y.stride(0),
            S1=1.2399, S2=1.1622,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
