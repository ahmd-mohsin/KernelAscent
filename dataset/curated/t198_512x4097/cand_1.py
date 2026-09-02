import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 198
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # load matmul output row (bf16), relu
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + eps)
    y = (x * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = (y * w).to(tl.bfloat16).to(tl.float32)

    # softmax #1 (fp32 accumulate, bf16 output like torch)
    p = tl.where(mask, p, float('-inf'))
    m1 = tl.max(p, axis=0)
    e1 = tl.math.exp(p - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # softmax #2
    p = tl.where(mask, p, float('-inf'))
    m2 = tl.max(p, axis=0)
    e2 = tl.math.exp(p - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    # relu (softmax output is non-negative; kept for exactness, no-op cost)
    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK_N = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            h2, self.rms2_w, out,
            N, h2.stride(0), out.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out.reshape(orig_shape)
