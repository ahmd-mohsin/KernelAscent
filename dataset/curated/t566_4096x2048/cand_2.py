import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 566
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_rms_gelu_kernel(
    X_ptr, W_ptr, B2_ptr, B4_ptr, Y_ptr,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, matching reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    x = (x * inv).to(tl.float16).to(tl.float32)

    # * rms1_w  (PyTorch half binary ops use fp32 opmath, round to fp16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.float16).to(tl.float32)

    # + b2
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b2).to(tl.float16).to(tl.float32)

    # gelu (exact erf, fp32 opmath, round to fp16)
    SQRT1_2: tl.constexpr = 0.7071067811865476
    x = (x * 0.5 * (1.0 + tl.math.erf(x * SQRT1_2))).to(tl.float16).to(tl.float32)

    # + b4
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b4).to(tl.float16).to(tl.float32)

    # gelu
    y = (x * 0.5 * (1.0 + tl.math.erf(x * SQRT1_2))).to(tl.float16)

    tl.store(Y_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_kernel[(rows,)](
            x, self.rms1_w, self.b2, self.b4, y,
            x.stride(0),
            N=N,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
