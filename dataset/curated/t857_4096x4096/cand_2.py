import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 857
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_ln_softmax3_kernel(
    X_ptr, G_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # round to fp16 (op boundary)
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax 1 ----
    y = tl.where(mask, y, float('-inf'))
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = e1 / s1
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax 2 ----
    y2 = tl.where(mask, y, float('-inf'))
    m2 = tl.max(y2, axis=0)
    e2 = tl.exp(y2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y = e2 / s2
    y = y.to(tl.float16).to(tl.float32)

    # ---- Scale (fp16 multiply like PyTorch) ----
    y = (y.to(tl.float16) * scale.to(tl.float16)).to(tl.float32)

    # ---- Softmax 3 ----
    y3 = tl.where(mask, y, float('-inf'))
    m3 = tl.max(y3, axis=0)
    e3 = tl.exp(y3 - m3)
    e3 = tl.where(mask, e3, 0.0)
    s3 = tl.sum(e3, axis=0)
    y = e3 / s3

    tl.store(Out_ptr + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_ln_softmax3_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, out,
            N, h.stride(0), out.stride(0),
            1e-5, 1.3209,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
