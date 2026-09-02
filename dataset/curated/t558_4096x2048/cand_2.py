import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 558
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_scale_bias_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # Replicate fp16 rounding of the reference elementwise ops
    x = (x.to(tl.float32) * S1).to(tl.float16)
    x = (x + b).to(tl.float16)
    x = (x.to(tl.float32) * S2).to(tl.float16)

    xf = tl.where(mask, x.to(tl.float32), float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out_ptr + row * stride_o + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 512 else 4
        _fused_scale_bias_softmax[(Mrows,)](
            y, self.b2, out,
            N, y.stride(0), out.stride(0),
            1.4099, 1.4586,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
