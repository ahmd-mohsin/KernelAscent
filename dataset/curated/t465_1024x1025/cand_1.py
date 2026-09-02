import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 465
M, D, DT = 1024, 1025, torch.float16


@triton.jit
def _scaled_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    # replicate reference: multiply by scale in fp16, then softmax in fp32
    s16 = tl.full((), 1.0, tl.float16) * scale
    x = (x * s16.to(x.dtype)).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM, fused in-place ReLU
        h = torch.matmul(x, self.W0)
        h.relu_()
        z = torch.matmul(h, self.W2)

        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _scaled_softmax_kernel[(m,)](
            z, out,
            n, z.stride(0), out.stride(0),
            1.0685,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
