import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 129
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_relu_softmax_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # relu
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))

    # softmax (fp32 accumulation, output rounded to bf16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y_bf16 = (e / s).to(tl.bfloat16)

    # RMSNorm on the bf16-rounded softmax output (computed in fp32)
    yf = y_bf16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    normed = (yf * inv).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (normed.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        out = torch.empty((Mrows, N), device=h.device, dtype=torch.bfloat16)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_softmax_rms_kernel[(Mrows,)](
            h, self.rms3_w, out,
            h.stride(0),
            N=N,
            BLOCK=BLOCK,
            EPS=1e-6,
            num_warps=8,
        )
        return out
