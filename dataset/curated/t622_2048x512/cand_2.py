import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 622
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU computed in fp32, rounded to bf16 (matches PyTorch)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    n = (g * inv).to(tl.bfloat16).to(tl.float32)

    # weight multiply (rounded to bf16, matching PyTorch elementwise mul)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = (n * w).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, output rounded to bf16
    v = tl.where(mask, v, float('-inf'))
    mx = tl.max(v, axis=0)
    e = tl.exp(v - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # final scalar multiply (fp32 opmath, rounded to bf16)
    out = (sm * scale).to(tl.bfloat16)
    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_rms_softmax_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N, h.stride(0), out.stride(0),
            1.0258,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
