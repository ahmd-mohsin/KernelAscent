import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 68
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_softmax2_rms_bias_gelu(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)

    # ---- softmax #1 (fp32 compute, fp16 rounding like PyTorch half softmax) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- softmax #2 ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32), round to fp16 ----
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)

    # ---- scale by weight (opmath fp32, round fp16) ----
    w = tl.load(W_ptr + offs).to(tl.float32)
    x = (x * w).to(tl.float16).to(tl.float32)

    # ---- add bias (opmath fp32, round fp16) ----
    b = tl.load(B_ptr + offs).to(tl.float32)
    x = (x + b).to(tl.float16).to(tl.float32)

    # ---- exact (erf) GELU in fp32 ----
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        rows, cols = h.shape  # cols == 2048
        y = torch.empty_like(h)

        _fused_softmax2_rms_bias_gelu[(rows,)](
            h, self.rms3_w, self.b4, y,
            h.stride(0), y.stride(0),
            N=cols,
            BLOCK=cols,
            num_warps=8,
        )
        return y
