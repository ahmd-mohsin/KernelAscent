import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 135
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_relu_rms_softmax_rms_relu(
    X_ptr, W2_ptr, W4_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row (fp16) and apply relu
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)

    # RMSNorm #1 (compute in fp32, cast to fp16, multiply by fp16 weight)
    xf = x.to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / N
    inv1 = tl.math.rsqrt(ms1 + 1e-6)
    y16 = (xf * inv1).to(tl.float16)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0)
    y16 = y16 * w2

    # Softmax (fp32 accumulation, fp16 output — matches PyTorch half softmax)
    z = y16.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    mx = tl.max(z, axis=0)
    e = tl.exp(z - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)

    # RMSNorm #2
    pf = p16.to(tl.float32)
    ms2 = tl.sum(pf * pf, axis=0) / N
    inv2 = tl.math.rsqrt(ms2 + 1e-6)
    q16 = (pf * inv2).to(tl.float16)
    w4 = tl.load(W4_ptr + offs, mask=mask, other=0.0)
    q16 = q16 * w4

    # final relu
    q16 = tl.maximum(q16, 0.0)

    tl.store(Out_ptr + row * stride_o + offs, q16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_relu_rms_softmax_rms_relu[(rows,)](
            h, self.rms2_w, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
