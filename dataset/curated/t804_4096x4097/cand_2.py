import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 804
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _rms_scale_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    N, stride_x,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X_ptr + row * stride_x + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    # match reference: (x * rsqrt).to(bf16) * w  + b  (both in bf16 arithmetic)
    xn = (x * inv_rms).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    y = xn * w + b

    tl.store(Out_ptr + row * stride_x + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _rms_scale_bias_kernel[(m,)](
            x, self.rms1_w, self.b2, out,
            n, x.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
