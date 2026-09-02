import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 722
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_softmax2_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax #1 (fp32 compute, round to bf16 like PyTorch)
    m1 = tl.max(x, axis=0)
    e1 = tl.math.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # softmax #2
    y_in = tl.where(mask, y, float('-inf'))
    m2 = tl.max(y_in, axis=0)
    e2 = tl.math.exp(y_in - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    z = (e2 / s2).to(tl.bfloat16).to(tl.float32)

    # RMSNorm (fp32 compute, round to bf16, then bf16*bf16 weight mult via fp32 opmath)
    zz = tl.where(mask, z * z, 0.0)
    ms = tl.sum(zz, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    normed = (z * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    res = (normed * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # (M, 1024) bf16 GEMM via cuBLAS
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax2_rms_kernel[(Mrows,)](
            h, self.rms3_w, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out @ self.W4
