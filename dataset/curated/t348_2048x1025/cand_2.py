import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 348
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _bias_relu_kernel(X_ptr, B_ptr, n_elem, N_COLS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + (offs % N_COLS), mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x + b, 0.0)
    tl.store(X_ptr + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _softmax_kernel(X_ptr, Y_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(X_ptr + row * N + offs).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS bf16, fp32 accumulate)
        h = x @ self.W0  # (M, 1024), contiguous

        # Fused bias-add + double-ReLU (relu(relu(z)) == relu(z)) in one pass
        n_elem = h.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elem, BLOCK),)
        _bias_relu_kernel[grid](h, self.b1, n_elem, N_COLS=1024, BLOCK=BLOCK)

        # GEMM 2
        z = h @ self.W4  # (M, 512), contiguous

        # Fused row softmax in a single kernel (fp32 compute, bf16 out)
        out = torch.empty_like(z)
        _softmax_kernel[(z.shape[0],)](z, out, N=512)
        return out
