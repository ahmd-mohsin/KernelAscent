import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 182
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gelu_bias_kernel(X, B, Y, n_elements, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU computed in fp32, matching PyTorch's opmath behavior
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round to bf16 (matmul->gelu output dtype), then add bias in fp32, round to bf16
    g = g.to(tl.bfloat16).to(tl.float32)
    b = tl.load(B + (offs % N), mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16)
    tl.store(Y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = x @ self.W1
            x = F.gelu(x)
            x = x + self.b3
            return x

        x = x.contiguous()
        m, d = x.shape

        # Fused row-wise softmax (fp32 accumulate, bf16 out)
        s = torch.empty_like(x)
        BLOCK_S = triton.next_power_of_2(d)
        _softmax_kernel[(m,)](
            x, s, x.stride(0), s.stride(0), d,
            BLOCK=BLOCK_S, num_warps=16,
        )

        # cuBLAS tensor-core matmul (fp32 accumulation)
        h = s @ self.W1

        # Fused exact GELU + bias add (in-place on h)
        n = h.numel()
        ncols = h.shape[-1]
        BLOCK_E = 1024
        grid = (triton.cdiv(n, BLOCK_E),)
        _gelu_bias_kernel[grid](
            h, self.b3, h, n, ncols,
            BLOCK=BLOCK_E, num_warps=4,
        )
        return h
