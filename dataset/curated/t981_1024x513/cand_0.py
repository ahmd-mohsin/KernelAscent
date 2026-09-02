import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 981
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_softmax_bias_rms_kernel(
    X_ptr, B_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    ptr = X_ptr + row * N + offs

    # ---- softmax (computed in fp32, matching PyTorch bf16 softmax) ----
    x = tl.load(ptr).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16)

    # ---- x = x * 1.4911 (bf16 op, fp32 opmath) ----
    p = (p.to(tl.float32) * 1.4911).to(tl.bfloat16)

    # ---- x = x + b3 ----
    b = tl.load(B_ptr + offs).to(tl.float32)
    v = (p.to(tl.float32) + b).to(tl.bfloat16)

    # ---- RMSNorm in fp32 ----
    vf = v.to(tl.float32)
    ms = tl.sum(vf * vf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    y = (vf * inv).to(tl.bfloat16)

    # ---- * rms4_w ----
    w = tl.load(W_ptr + offs).to(tl.float32)
    y = (y.to(tl.float32) * w).to(tl.bfloat16)

    # ---- * 1.1793 ----
    y = (y.to(tl.float32) * 1.1793).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (same as reference)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        _fused_softmax_bias_rms_kernel[(rows,)](
            h, self.b3, self.rms4_w, out,
            N=N,
            num_warps=8,
        )
        return out
