import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 852
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _epilogue_softmax_kernel(
    X_ptr, B1_ptr, B5_ptr, OUT_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0)
    b5 = tl.load(B5_ptr + cols, mask=mask, other=0.0)

    # emulate bf16 rounding at each elementwise step for numerical equivalence
    v = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    v = tl.maximum(v, 0.0)
    v = (v.to(tl.float32) * 1.4375).to(tl.bfloat16)
    v = (v.to(tl.float32) * 1.0728).to(tl.bfloat16)
    v = (v.to(tl.float32) + b5.to(tl.float32)).to(tl.bfloat16)

    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    row_max = tl.max(vf, axis=0)
    e = tl.exp(vf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(OUT_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = h.shape
        out = torch.empty((m, n), device=h.device, dtype=torch.bfloat16)
        BLOCK = triton.next_power_of_2(n)
        _epilogue_softmax_kernel[(m,)](
            h, self.b1, self.b5, out,
            h.stride(0),
            N=n,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
