import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 406
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load one row (bf16 -> fp32)
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches torch.softmax acc-type behavior for bf16 inputs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    p = e / s

    # cast to bf16 first (reference materializes softmax output in bf16, then .float())
    p_bf16 = p.to(tl.bfloat16)
    pf = p_bf16.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, pf * pf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y_bf16 = (pf * r).to(tl.bfloat16)

    # multiply by weight (bf16 * bf16 computed at fp32 opmath, rounded to bf16)
    w = tl.load(W_ptr + offs, mask=mask).to(tl.float32)
    out = (y_bf16.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0  # (M, 512) bf16
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
