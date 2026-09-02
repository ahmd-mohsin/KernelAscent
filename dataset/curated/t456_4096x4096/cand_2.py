import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 456
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _relu_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # relu on valid entries, keep -inf on padding so it doesn't affect max
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two GEMMs via cuBLAS tensor cores (kept separate for exact numerics)
        h = x @ self.W0
        z = h @ self.W1

        if not z.is_cuda:
            return torch.softmax(torch.relu(z), dim=-1)

        z = z.contiguous()
        orig_shape = z.shape
        N = orig_shape[-1]
        z2 = z.view(-1, N)
        rows = z2.shape[0]
        out = torch.empty_like(z2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _relu_softmax_kernel[(rows,)](
            z2, out, N,
            z2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
