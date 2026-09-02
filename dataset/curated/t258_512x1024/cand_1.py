import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 258
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _bias_scale_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)  # bf16
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)            # bf16

    # Emulate PyTorch's per-op bf16 rounding:
    # (x + b) -> bf16, (* 1.2924) -> bf16, (* 1.4915) -> bf16
    v = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    v = (v.to(tl.float32) * 1.2924).to(tl.bfloat16)
    v = (v.to(tl.float32) * 1.4915).to(tl.bfloat16)

    vf = tl.where(mask, v.to(tl.float32), float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _bias_scale_softmax_kernel[(m,)](
            y, self.b1, out, n,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
