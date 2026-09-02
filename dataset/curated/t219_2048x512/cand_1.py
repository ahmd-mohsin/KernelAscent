import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 219
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _softmax_scale_kernel(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    p = num / denom

    # match reference: softmax result rounded to fp16, then scaled (fp32 opmath), rounded to fp16
    p16 = p.to(tl.float16)
    out = (p16.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y_ptr + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # fused (x @ W0 + b1) via cuBLAS addmm
        h = torch.addmm(self.b1, x, self.W0)
        z = torch.mm(h, self.W2)

        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _softmax_scale_kernel[(m,)](
            z, out,
            z.stride(0), out.stride(0),
            n,
            SCALE=1.481,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
