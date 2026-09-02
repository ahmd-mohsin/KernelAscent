import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 722
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _softmax2_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)

    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)

    # --- softmax 1 (fp32 math, output rounded to bf16 as in eager) ---
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    x = e / tl.sum(e, axis=0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # --- softmax 2 ---
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    x = e / tl.sum(e, axis=0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm (fp32) ---
    ms = tl.sum(x * x, axis=0) / N
    r = tl.rsqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)

    # --- scale by weight (fp32 opmath, round to bf16 like eager mul) ---
    w = tl.load(W_ptr + offs).to(tl.float32)
    y = (x * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS bf16 tensor cores)
        h = x @ self.W0
        if not h.is_contiguous():
            h = h.contiguous()

        m, n = h.shape  # n == 1024
        out = torch.empty_like(h)

        # Fused: softmax -> softmax -> RMSNorm -> * weight  (single pass over rows)
        _softmax2_rms_kernel[(m,)](
            h, self.rms3_w, out,
            h.stride(0), out.stride(0),
            N=n,
            num_warps=8,
            num_stages=1,
        )

        # GEMM 2
        return out @ self.W4
