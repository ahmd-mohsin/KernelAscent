import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 852
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B1_ptr, B5_ptr, Out_ptr,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Replicate PyTorch bf16 elementwise semantics:
    # each op computes in fp32 (opmath) then rounds back to bf16.
    v = (x + b1).to(tl.bfloat16).to(tl.float32)
    v = tl.maximum(v, 0.0)
    v = (v * 1.4375).to(tl.bfloat16).to(tl.float32)
    v = (v * 1.0728).to(tl.bfloat16).to(tl.float32)
    v = (v + b5).to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out_ptr + row * stride_row + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores, fp32 accumulate) - same as reference
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N_)
        _fused_epilogue_softmax[(M_,)](
            h, self.b1, self.b5, out,
            h.stride(0),
            N=N_,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
