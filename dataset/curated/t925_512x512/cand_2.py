import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 925
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_scale_bias_gelu_kernel(
    Y_ptr, B_ptr,
    n_elements, n_cols,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    y = tl.load(Y_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.045  (compute fp32, round to bf16 to match torch semantics)
    y = y * 1.045
    y = y.to(tl.bfloat16).to(tl.float32)

    # x = x + b2
    col = offs % n_cols
    b = tl.load(B_ptr + col, mask=mask, other=0.0).to(tl.float32)
    y = y + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    tl.store(Y_ptr + offs, g.to(tl.bfloat16), mask=mask)


@triton.jit
def _softmax_kernel(
    X_ptr, OUT_ptr,
    n_cols, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(OUT_ptr + row * stride_row + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS, TF32/BF16 tensor cores)
        h = x @ self.W0  # (M, 2048), bf16

        h = h.contiguous()
        n_elements = h.numel()
        n_cols = h.shape[-1]
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        _fused_scale_bias_gelu_kernel[grid](
            h, self.b2, n_elements, n_cols, BLOCK=BLOCK,
            num_warps=4,
        )

        # GEMM 2
        z = h @ self.W4  # (M, 1024), bf16
        z = z.contiguous()

        out = torch.empty_like(z)
        n_rows, n_cols2 = z.shape[-2], z.shape[-1]
        z2 = z.view(-1, n_cols2)
        out2 = out.view(-1, n_cols2)
        BLOCK_SM = triton.next_power_of_2(n_cols2)
        _softmax_kernel[(z2.shape[0],)](
            z2, out2, n_cols2, z2.stride(0), BLOCK=BLOCK_SM,
            num_warps=8,
        )
        return out
