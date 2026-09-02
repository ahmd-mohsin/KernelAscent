import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 103
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B1_ptr, B3_ptr, OUT_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Replicate PyTorch half elementwise semantics: compute in fp32,
    # round to fp16 after each op.
    x = (x + b1).to(tl.float16).to(tl.float32)
    x = (x * 1.1112).to(tl.float16).to(tl.float32)
    x = (x + b3).to(tl.float16).to(tl.float32)
    x = (x * 1.3457).to(tl.float16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # Softmax with fp32 accumulation (matches PyTorch half softmax)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(OUT_ptr + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM with fp32 accumulation (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_epilogue_softmax[(m,)](
            y, self.b1, self.b3, out,
            n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
