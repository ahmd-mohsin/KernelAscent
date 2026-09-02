import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 348
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _bias_relu_kernel(x_ptr, b_ptr, n_elements, N_COLS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    col = offs % N_COLS
    b = tl.load(b_ptr + col, mask=mask, other=0.0)
    y = x + b
    y = tl.maximum(y, 0.0)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    ptr = x_ptr + row * stride + cols
    x = tl.load(ptr, mask=mask, other=-float('inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(out_ptr + row * stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1
        h = x @ self.W0  # (M, 1024)
        h = h.contiguous()

        # Fused bias + relu (relu(relu(x)) == relu(x))
        n_elements = h.numel()
        n_cols = h.shape[-1]
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        _bias_relu_kernel[grid](h, self.b1, n_elements, N_COLS=n_cols, BLOCK=BLOCK)

        # GEMM 2
        z = h @ self.W4  # (M, 512)
        z = z.contiguous()

        # Fused softmax over last dim
        out = torch.empty_like(z)
        n_rows, sm_cols = z.shape
        SM_BLOCK = triton.next_power_of_2(sm_cols)
        _softmax_kernel[(n_rows,)](z, out, sm_cols, z.stride(0), BLOCK=SM_BLOCK,
                                   num_warps=8 if SM_BLOCK >= 512 else 4)
        return out
