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
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # Match reference bf16 rounding at each elementwise step
    t = (x.to(tl.float32) * 1.2399).to(tl.bfloat16)
    t = tl.maximum(t, 0.0)
    t = (t.to(tl.float32) * 1.1622).to(tl.bfloat16)

    # Softmax in fp32 (as PyTorch does internally for bf16 inputs)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    row_max = tl.max(tf, axis=0)
    e = tl.exp(tf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS/tensor-core matmul (bf16)
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_scale_relu_softmax[(M_,)](
            h, y,
            h.stride(0), y.stride(0),
            N_, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
