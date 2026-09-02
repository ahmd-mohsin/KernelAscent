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
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_bf16 = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0)
    x = x_bf16.to(tl.float32)

    # x = x * 1.2974  (computed in fp32, rounded to bf16 like PyTorch opmath)
    x = (x * 1.2974).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    x = (x * inv).to(tl.bfloat16).to(tl.float32)

    # * rms2_w (bf16 mul, opmath fp32, round bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.bfloat16).to(tl.float32)

    # * 1.1325 (round bf16)
    x = (x * 1.1325).to(tl.bfloat16).to(tl.float32)

    # exact GELU in fp32
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(Y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_rms_gelu_kernel[(m,)](
            h, self.rms2_w, out,
            N=n, BLOCK_N=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
