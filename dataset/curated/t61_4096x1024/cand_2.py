import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 61
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _scale_bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # replicate bf16 rounding of: (x * scale) then (+ bias), as in reference
    t = (x * SCALE).to(tl.bfloat16).to(tl.float32)
    t = (t + b).to(tl.bfloat16).to(tl.float32)

    t = tl.where(mask, t, float('-inf'))
    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS/tensor cores) + in-place ReLU (no extra allocation)
        h = torch.mm(x, self.W0)
        torch.relu_(h)

        # GEMM 2
        z = torch.mm(h, self.W2)

        # Fused scale + bias + softmax in a single Triton kernel
        Mrows, N = z.shape
        out = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _scale_bias_softmax_kernel[(Mrows,)](
            z, self.b4, out,
            N, z.stride(0), out.stride(0),
            SCALE=1.1673,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
