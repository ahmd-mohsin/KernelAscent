import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 945
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _softmax_bias_gelu_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax which accumulates in fp32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # round to fp16 (softmax output dtype), like eager mode does
    p16 = p.to(tl.float16)

    # bias add: fp16 tensors, computed in fp32 opmath, rounded back to fp16
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t16 = (p16.to(tl.float32) + b).to(tl.float16)

    # exact GELU (erf) computed in fp32 opmath, rounded to fp16
    tf = t16.to(tl.float32)
    g = 0.5 * tf * (1.0 + tl.math.erf(tf * 0.7071067811865476))

    tl.store(Out_ptr + row * stride_o + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = x @ self.W0

        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _softmax_bias_gelu_kernel[(m,)](
            y, self.b2, out,
            n, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
