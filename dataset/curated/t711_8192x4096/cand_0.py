import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 711
M, D, DT = 8192, 4096, torch.bfloat16


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

    # softmax (fp32 accumulation, matching PyTorch's bf16 softmax which computes in fp32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # round to bf16 (softmax output dtype), then relu
    p = p.to(tl.bfloat16).to(tl.float32)
    p = tl.maximum(p, 0.0)

    # add bias in fp32, round to bf16 (matches bf16 tensor add with fp32 opmath)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (p + b).to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU in fp32
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_y + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        y = x @ self.W0  # (M, 1024) bf16

        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _softmax_relu_bias_gelu_kernel[(Mrows,)](
            y, self.b3, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
