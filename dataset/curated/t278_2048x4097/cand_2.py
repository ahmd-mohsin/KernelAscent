import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 278
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_epilogue_rmsnorm(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    scale,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # relu (idempotent, applied once)
    x = tl.maximum(x, 0.0)

    # x * 1.3017 computed in fp32, rounded to bf16 (matches PyTorch opmath)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    # x + b4 computed in fp32, rounded to bf16
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul on tensor cores
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_epilogue_rmsnorm[(m,)](
            h, self.b4, self.rms5_w, out,
            n, h.stride(0), out.stride(0),
            1.3017, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=16,
        )
        return out
