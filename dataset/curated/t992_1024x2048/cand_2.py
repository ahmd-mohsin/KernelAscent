import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 992
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_relu_bias_gelu_rms_kernel(
    X_ptr, B_ptr, W_ptr, Y_ptr,
    D: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # relu (exact in fp16) then bias add computed in fp32, rounded to fp16
    x = tl.maximum(x, 0.0)
    x = x + b
    x = x.to(tl.float16).to(tl.float32)

    # exact (erf) GELU computed in fp32, rounded to fp16 (matches PyTorch half gelu)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(g * g, axis=0) / D
    inv = tl.math.rsqrt(ms + eps)
    y = (g * inv).to(tl.float16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.float16)
    tl.store(Y_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        h = h.contiguous()
        rows, d = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(d)
        _fused_relu_bias_gelu_rms_kernel[(rows,)](
            h, self.b2, self.rms4_w, out,
            d, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
