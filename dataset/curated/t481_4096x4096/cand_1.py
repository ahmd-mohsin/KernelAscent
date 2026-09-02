import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 481
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_relu_rms_relu_softmax(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load one row (bf16) -> relu -> float32 for RMS reduction
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # RMS norm in fp32 (mean of squares over the row)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.rsqrt(ms + EPS)

    # cast normalized value back to bf16, multiply by bf16 weight (matches reference dtype flow)
    xn = (xf * inv).to(tl.bfloat16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    y = xn * w
    y = tl.maximum(y, 0.0)

    # softmax with fp32 accumulation (matches PyTorch's bf16 softmax path)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_relu_rms_relu_softmax[(Mrows,)](
            h, self.rms2_w, out,
            N, h.stride(0), out.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
