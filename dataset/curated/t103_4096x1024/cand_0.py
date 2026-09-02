import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 103
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_bias_scale_relu_softmax(
    X_ptr, B1_ptr, B3_ptr, Out_ptr,
    N, stride_xm, stride_om,
    S1, S2,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0)

    # replicate fp16 rounding at each elementwise step (opmath in fp32, round to fp16)
    v = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    v = (v.to(tl.float32) * S1).to(tl.float16)
    v = (v.to(tl.float32) + b3.to(tl.float32)).to(tl.float16)
    v = (v.to(tl.float32) * S2).to(tl.float16)
    v = tl.maximum(v, 0.0)

    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    row_max = tl.max(vf, axis=0)
    num = tl.exp(vf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = (num / denom).to(tl.float16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS tensor-core GEMM
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_bias_scale_relu_softmax[(m,)](
            y, self.b1, self.b3, out,
            n, y.stride(0), out.stride(0),
            1.1112, 1.3457,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
