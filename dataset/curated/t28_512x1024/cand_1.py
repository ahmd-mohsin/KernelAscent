import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 28
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_scale_rms_softmax(
    X_ptr, W_ptr, Out_ptr,
    stride_xm, stride_om,
    N, scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    # scale in fp16 (replicates reference rounding)
    x = (x.to(tl.float32) * scale).to(tl.float16)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # fp16
    y16 = (xf * inv).to(tl.float16) * w  # fp16 multiply like reference
    yf = y16.to(tl.float32)

    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_rms_softmax[(Mrows,)](
            h, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N, 1.1067, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
