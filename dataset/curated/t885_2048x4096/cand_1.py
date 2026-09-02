import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 885
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_softmax2_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # First softmax (float accumulation, output rounded to fp16 like torch)
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, 0)
    p = (e1 / s1).to(tl.float16)

    # x * 1.3127 (opmath float, result fp16)
    t = (p.to(tl.float32) * 1.3127).to(tl.float16)

    # + bias (opmath float, result fp16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    u = (t.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # Second softmax
    v = tl.where(mask, u.to(tl.float32), float('-inf'))
    m2 = tl.max(v, 0)
    e2 = tl.exp(v - m2)
    s2 = tl.sum(e2, 0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax2_kernel[(Mrows,)](
            h, self.b3, y,
            h.stride(0), y.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
