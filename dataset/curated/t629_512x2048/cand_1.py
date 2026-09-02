import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 629
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _softmax_rms_bias_kernel(
    X_ptr, W_ptr, B3_ptr, B4_ptr, Out_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, as PyTorch does for fp16 inputs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (softmax output dtype), then back to fp32 for RMSNorm
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)

    ms = tl.sum(pf * pf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    y = (pf * r).to(tl.float16)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    y = y * w
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0)
    y = y + b3
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0)
    y = y + b4

    tl.store(Out_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _softmax_rms_bias_kernel[(rows,)](
            h, self.rms2_w, self.b3, self.b4, out,
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
