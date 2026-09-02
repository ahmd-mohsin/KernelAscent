import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 106
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _bias_add_kernel(X, B, OUT, n_elements, D_: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask)
    b = tl.load(B + (offs % D_), mask=mask)
    tl.store(OUT + offs, x + b, mask=mask)


@triton.jit
def _softmax_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = x @ self.W1
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        Mrows, Dcols = x.shape

        # Fused bias add (elementwise, single pass)
        h = torch.empty_like(x)
        n = x.numel()
        BLK = 1024
        _bias_add_kernel[(triton.cdiv(n, BLK),)](
            x, self.b0, h, n, D_=Dcols, BLOCK=BLK, num_warps=4
        )

        # Matmul via cuBLAS (tensor cores on A100 for bf16)
        z = h @ self.W1

        # Fused softmax over last dim (one program per row, in-place store)
        N = z.shape[1]
        BLOCK = triton.next_power_of_2(N)
        _softmax_kernel[(z.shape[0],)](
            z, z, z.stride(0), z.stride(0), N, BLOCK=BLOCK, num_warps=8
        )
        return z
