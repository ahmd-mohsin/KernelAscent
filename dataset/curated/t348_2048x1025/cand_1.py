import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 348
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self._has_fused_addmm = hasattr(torch, "_addmm_activation")

    def forward(self, x):
        # Fused GEMM + bias + ReLU via cuBLASLt epilogue (relu(relu(x)) == relu(x))
        if self._has_fused_addmm and x.is_cuda:
            h = torch._addmm_activation(self.b1, x, self.W0)
        else:
            h = torch.addmm(self.b1, x, self.W0)
            h.relu_()

        # Second GEMM (cuBLAS tensor cores)
        z = h @ self.W4

        if not z.is_cuda:
            return torch.softmax(z, dim=-1)

        # Fused row-wise softmax in a single Triton kernel
        z = z.contiguous()
        out = torch.empty_like(z)
        n_rows, n_cols = z.shape
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _softmax_kernel[(n_rows,)](
            z, out,
            z.stride(0), out.stride(0),
            n_cols,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
