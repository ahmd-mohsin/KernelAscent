import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 326
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)  # bf16
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)                   # bf16

    # relu
    x = tl.maximum(x, 0.0)
    # x * 1.0573 (fp32 math, round to bf16) -- matches PyTorch bf16 elementwise
    x = (x.to(tl.float32) * 1.0573).to(tl.bfloat16)
    # x * 1.4239
    x = (x.to(tl.float32) * 1.4239).to(tl.bfloat16)
    # second relu is a no-op after relu+positive scales, but keep for safety
    x = tl.maximum(x, 0.0)
    # x + b5 (fp32 math, round to bf16)
    x = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    den = tl.sum(num, axis=0)
    y = (num / den).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = x @ self.W0  # (M, 512) bf16

        if not y.is_cuda:
            # CPU fallback: reference path
            z = torch.relu(y)
            z = z * 1.0573
            z = z * 1.4239
            z = torch.relu(z)
            z = z + self.b5
            return torch.softmax(z, dim=-1)

        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 512 else 4
        _fused_epilogue_softmax[(Mrows,)](
            y, self.b5, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
