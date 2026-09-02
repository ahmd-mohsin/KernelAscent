import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 716
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU, computed in fp32 like PyTorch does for fp16 inputs
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _bias_softmax_kernel(X, B, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    # fp16 add of bias to exactly match (x @ W1).half() + b2 in fp16
    x = tl.load(X + row * D + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    v = (x + b).to(tl.float32)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape

        # Fused elementwise GELU (fp32 internal math, fp16 output)
        gx = torch.empty_like(x)
        n = x.numel()
        BLOCK_E = 1024
        _gelu_kernel[(triton.cdiv(n, BLOCK_E),)](x, gx, n, BLOCK=BLOCK_E, num_warps=4)

        # cuBLAS tensor-core GEMM (fp16 in/out, fp32 accumulate)
        y = gx @ self.W1

        # Fused bias-add + row softmax (bias add in fp16, softmax in fp32)
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(D_)
        _bias_softmax_kernel[(M_,)](y, self.b2, out, D_, BLOCK=BLOCK, num_warps=16)
        return out
