import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 30
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_bias_softmax_rms_kernel(
    Y_ptr, B_ptr, W_ptr, OUT_ptr,
    N, stride_y, stride_o,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul result (fp16) and bias (fp16), add in fp16 (matches x + b1 in half)
    y = tl.load(Y_ptr + row * stride_y + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    x16 = y + b

    # softmax in fp32, output rounded to fp16 (matches torch.softmax on half)
    xf = x16.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm16 = (e / s).to(tl.float16)

    # x * 1.0129 in fp32 opmath, rounded back to fp16
    sc16 = (sm16.to(tl.float32) * scale).to(tl.float16)

    # RMSNorm in fp32
    xf2 = sc16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf2 * xf2, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    normed16 = (xf2 * inv).to(tl.float16)

    # multiply by weight (half*half done in fp32 opmath, rounded to fp16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = (normed16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(OUT_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 tensor-core matmul
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_bias_softmax_rms_kernel[(Mrows,)](
            y, self.b1, self.rms4_w, out,
            N, y.stride(0), out.stride(0),
            1.0129, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
