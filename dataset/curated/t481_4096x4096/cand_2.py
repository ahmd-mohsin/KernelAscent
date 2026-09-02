import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 481
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_rms_relu_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_bf = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    x = x_bf.to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # RMSNorm (fp32 accumulation), matching: (xf * rsqrt(mean(xf^2)+eps)).to(bf16) * w
    msq = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(msq + EPS)
    y = (x * r).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = y * w  # bf16 multiply, same as reference

    # relu (bf16)
    zero_bf = tl.zeros([BLOCK], dtype=tl.bfloat16)
    y = tl.maximum(y, zero_bf)

    # softmax in fp32 (matches PyTorch's fp32-accumulated softmax on bf16)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_relu_softmax_kernel[(Mrows,)](
            h, self.rms2_w, out,
            h.stride(0),
            N=N,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
