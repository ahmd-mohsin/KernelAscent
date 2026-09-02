import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 966
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # relu (bf16, exact)
    x = tl.maximum(x, 0.0)
    # scale: compute in fp32, round to bf16 (matches PyTorch scalar op semantics)
    x = (x.to(tl.float32) * SCALE).to(tl.bfloat16)
    # add bias: compute in fp32, round to bf16
    x = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    # relu
    x = tl.maximum(x, 0.0)

    # softmax in fp32 accumulation (matches PyTorch bf16 softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0

        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_epilogue_softmax[(m,)](
            h, self.b3, out,
            n, h.stride(0), out.stride(0),
            SCALE=1.1429,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
