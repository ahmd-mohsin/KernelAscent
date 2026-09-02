import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 182
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gelu_bias_kernel(X, B, Y, n_elements, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU computed in fp32, matching PyTorch's opmath for bf16
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round to bf16 (as F.gelu output would be), then add bias in fp32 opmath
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
            # CPU fallback (reference path)
            x = torch.softmax(x, dim=-1)
            x = x @ self.W1
            x = F.gelu(x)
            return x + self.b3

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).contiguous()
        M_, D_ = x2.shape

        # Fused softmax (single pass per row) in Triton
        s = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(D_)
        _softmax_kernel[(M_,)](
            x2, s, D_, x2.stride(0), s.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )

        # Matmul via cuBLAS (bf16 in, fp32 accumulate)
        y = s @ self.W1
        y = y.contiguous()

        # Fused GELU + bias add in one elementwise kernel
        out = torch.empty_like(y)
        n = y.numel()
        N_ = y.shape[-1]
        EBLOCK = 1024
        grid = (triton.cdiv(n, EBLOCK),)
        _gelu_bias_kernel[grid](y, self.b3, out, n, N_, BLOCK=EBLOCK, num_warps=4)

        return out.reshape(*orig_shape[:-1], N_)
