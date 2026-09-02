import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 198
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_relu_rms_softmax2_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # rmsnorm in fp32 (matching reference: mean of squares over full row)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xb = (x * r).to(tl.bfloat16)

    # multiply by weight in bf16 (matches: x.to(dtype) * rms2_w)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    xb = xb * w
    xf = xb.to(tl.float32)

    # softmax #1 (fp32 accumulation, output cast to bf16 like torch)
    xf = tl.where(mask, xf, float('-inf'))
    m1 = tl.max(xf, axis=0)
    e1 = tl.exp(xf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # softmax #2
    y1 = tl.where(mask, y1, float('-inf'))
    m2 = tl.max(y1, axis=0)
    e2 = tl.exp(y1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = e2 / s2

    # relu (no-op after softmax, kept for exactness)
    y2 = tl.maximum(y2, 0.0)

    tl.store(Y_ptr + row * N + offs, y2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_rms_softmax2_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
