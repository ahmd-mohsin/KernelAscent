import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 672
M, D, DT = 512, 513, torch.float16


@triton.jit
def _relu_bias_kernel(
    X_ptr, B_ptr,
    n_elements, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + (offs % N), mask=mask, other=0.0)
    # relu in fp16 then add bias in fp16 (matches reference order/precision)
    y = tl.maximum(x, 0.0) + b
    tl.store(X_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores) -> bit-identical to reference matmul
        h = torch.matmul(x, self.W0)

        if h.is_cuda:
            # Fused ReLU + bias-add in a single elementwise Triton kernel
            h = h.contiguous()
            n_elements = h.numel()
            N = h.shape[-1]
            BLOCK = 1024
            grid = (triton.cdiv(n_elements, BLOCK),)
            _relu_bias_kernel[grid](h, self.b2, n_elements, N, BLOCK=BLOCK)
        else:
            h = torch.relu(h)
            h = h + self.b2

        # GEMM 2 (cuBLAS tensor cores)
        out = torch.matmul(h, self.W3)

        # Fused LayerNorm (single ATen kernel, identical numerics to reference)
        out = F.layer_norm(out, (out.shape[-1],), self.ln4_g, self.ln4_b)
        return out
