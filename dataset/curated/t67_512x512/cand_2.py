import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 67
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _relu_scale_kernel(X_ptr, N, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # relu -> multiply by positive scale (second relu is a no-op since scale > 0)
    x = tl.maximum(x, 0.0)
    y = (x.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(X_ptr + offs, y, mask=mask)


@triton.jit
def _bias_softmax_kernel(Y_ptr, B_ptr, O_ptr, N, stride_y, stride_o,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    y = tl.load(Y_ptr + row * stride_y + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # bf16 add with fp32 opmath, rounded back to bf16 (matches eager `x + b5`)
    z_bf16 = (y.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    z = z_bf16.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))

    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(O_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS, tensor cores)
        h = torch.matmul(x, self.W0)

        # Fused relu + scale (+ redundant relu) elementwise, in-place
        n = h.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _relu_scale_kernel[grid](h, n, 1.3219, BLOCK=BLOCK)

        # GEMM 2 (cuBLAS, tensor cores)
        y = torch.matmul(h, self.W4)

        # Fused bias-add + softmax (one row per program)
        out = torch.empty_like(y)
        rows, cols = y.shape
        BLOCK_N = triton.next_power_of_2(cols)
        _bias_softmax_kernel[(rows,)](
            y, self.b5, out, cols, y.stride(0), out.stride(0),
            BLOCK=BLOCK_N,
        )
        return out
