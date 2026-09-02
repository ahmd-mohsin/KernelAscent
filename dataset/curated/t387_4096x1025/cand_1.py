import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 387
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _bias_scale_rmsnorm_kernel(
    X, B, W, Out,
    N, stride_x, stride_o,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b  (PyTorch half add: compute in fp32, round to fp16)
    x = (x + b).to(tl.float16).to(tl.float32)
    # x = x * 1.0435 (compute in fp32, round to fp16)
    x = (x * scale).to(tl.float16).to(tl.float32)

    # RMSNorm in float32
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    rs = 1.0 / tl.sqrt(ms + eps)
    y = (x * rs).to(tl.float16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.float16)
    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS half GEMM
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _bias_scale_rmsnorm_kernel[(Mrows,)](
            x, self.b1, self.rms3_w, out,
            N, x.stride(0), out.stride(0),
            1e-6, 1.0435,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
