import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 731
M, D, DT = 512, 512, torch.float16


@triton.jit
def _softmax_relu_bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # relu (no-op on softmax output but kept for exactness)
    p = tl.maximum(p, 0.0)

    # cast to fp16 (softmax output dtype), add bias in fp16 (matches x + b3 in half)
    p16 = p.to(tl.float16)
    b16 = tl.load(B_ptr + offs, mask=mask, other=0.0)
    t16 = p16 + b16

    # exact (erf) gelu computed in fp32, like PyTorch opmath for half
    t = t16.to(tl.float32)
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_y + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 4096) fp16

        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_relu_bias_gelu_kernel[(Mrows,)](
            h, self.b3, y,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2
        return y @ self.W5
