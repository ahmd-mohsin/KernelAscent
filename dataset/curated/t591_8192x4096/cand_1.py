import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 591
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_bias_relu_bias_softmax(
    X_ptr, B1_ptr, B3_ptr, Out_ptr,
    stride_x, stride_o,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0)

    # Emulate bf16 arithmetic exactly: fp32 add of two bf16 values is exact,
    # then round back to bf16 (matches bf16 "+" in eager PyTorch).
    t = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    t = tl.maximum(t, 0.0).to(tl.bfloat16)  # relu is exact in bf16
    t = (t.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)

    # Softmax computed in fp32 (matches PyTorch bf16 softmax behavior).
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float("-inf"))
    m = tl.max(tf, axis=0)
    e = tl.exp(tf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (same numerics as reference matmul)
        h = x @ self.W0

        M_rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_bias_relu_bias_softmax[(M_rows,)](
            h, self.b1, self.b3, out,
            h.stride(0), out.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
