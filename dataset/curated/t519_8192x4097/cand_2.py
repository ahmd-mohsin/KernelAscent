import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 519
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _softmax_scale_rmsnorm_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load one row of the matmul output, upcast to fp32 (matches PyTorch softmax internals)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # Softmax in fp32
    row_max = tl.max(x, axis=0)
    e = tl.math.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # Cast to bf16 (softmax output dtype), then scale by 1.0722 in fp32 opmath, round to bf16
    p_bf16 = p.to(tl.bfloat16)
    y = (p_bf16.to(tl.float32) * 1.0722).to(tl.bfloat16)

    # RMSNorm: upcast to fp32
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)

    # Normalize, round to bf16, then multiply by weight (fp32 opmath), round to bf16
    normed = (yf * r).to(tl.bfloat16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (normed.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Heavy GEMM: use cuBLAS (tensor cores on A100)
        h = x @ self.W0  # (M, 1024), bf16
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _softmax_scale_rmsnorm_kernel[(Mrows,)](
            h, self.rms3_w, out,
            h.stride(0), out.stride(0),
            N=N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
