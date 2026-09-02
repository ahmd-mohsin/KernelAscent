import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 635
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_rms_gelu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    S1: tl.constexpr,   # 1.2974
    S2: tl.constexpr,   # 1.1325
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load bf16 row, promote to fp32 (matches PyTorch opmath)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.2974   (fp32 compute, round to bf16 like PyTorch)
    x = (x * S1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    r = tl.rsqrt(ms + EPS)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)

    # * rms2_w (bf16 elementwise mul -> fp32 opmath, round bf16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xw = (xn * w).to(tl.bfloat16).to(tl.float32)

    # * 1.1325
    xs = (xw * S2).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based) in fp32, cast to bf16
    SQRT1_2: tl.constexpr = 0.7071067811865476
    y = xs * 0.5 * (1.0 + tl.math.erf(xs * SQRT1_2))

    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # bf16 GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N, 1.2974, 1.1325, 1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
