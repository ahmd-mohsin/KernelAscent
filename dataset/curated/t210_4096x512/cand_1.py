import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 210
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _gelu_softmax_kernel(
    X_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, computed in fp32 like PyTorch's opmath for bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round intermediate to bf16 to match reference (gelu output is materialized in bf16)
    g = g.to(tl.bfloat16).to(tl.float32)

    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # scale then matmul (tensor cores handle the GEMM)
        h = torch.matmul(x * 1.02, self.W1)
        h = h.contiguous()

        rows, N = h.shape[0] * (h.numel() // (h.shape[-1] * h.shape[0])) if h.dim() > 2 else h.shape[0], h.shape[-1]
        h2 = h.view(-1, N)
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _gelu_softmax_kernel[(h2.shape[0],)](
            h2, out,
            N,
            h2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view_as(h)
