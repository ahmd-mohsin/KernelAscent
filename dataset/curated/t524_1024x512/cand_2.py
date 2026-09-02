import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 524
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B_ptr, Out_ptr,
    stride_xm, stride_om,
    N: tl.constexpr,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # Emulate bf16 elementwise ops with fp32 opmath + round-to-bf16 (PyTorch semantics)
    v = x.to(tl.float32)
    v = (v * S1).to(tl.bfloat16).to(tl.float32)
    v = (v * S2).to(tl.bfloat16).to(tl.float32)
    v = (v + b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    v = tl.maximum(v, 0.0)

    # Softmax in fp32 (matches PyTorch bf16 softmax accumulation)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 matmul, same as reference
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_epilogue_softmax[(Mrows,)](
            y, self.b3, out,
            y.stride(0), out.stride(0),
            N, 1.2872, 1.4411,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
