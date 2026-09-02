import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 890
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _rmsnorm_softmax_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    normed = xf * inv

    # cast to bf16, multiply by weight in bf16 (matches reference)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y_bf16 = (normed.to(tl.bfloat16) * w)

    # softmax in fp32 (matches torch's internal fp32 accumulation for bf16)
    yf = y_bf16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS matmul (bf16 tensor cores)
        Mrows, N = x.shape
        x = x.contiguous()
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_softmax_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
