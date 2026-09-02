import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 465
M, D, DT = 1024, 1025, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    # replicate: (x * 1.0685) computed in fp16, then softmax in fp32
    xs = (x.to(tl.float32) * SCALE).to(tl.float16).to(tl.float32)
    xs = tl.where(mask, xs, float('-inf'))

    m = tl.max(xs, axis=0)
    e = tl.exp(xs - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores) + in-place ReLU
        h = torch.mm(x, self.W0)
        h.relu_()
        # GEMM 2
        z = torch.mm(h, self.W2)
        # Fused scale + softmax in one Triton kernel
        rows, cols = z.shape
        out = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _scale_softmax_kernel[(rows,)](
            z, out,
            cols, z.stride(0), out.stride(0),
            SCALE=1.0685,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
