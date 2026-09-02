import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 815
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _rms_scale_relu_kernel(
    X_ptr, W_ptr, Out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # mean of squares over the row
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # normalize, round to bf16 (matches .to(x.dtype))
    y = (x * inv).to(tl.bfloat16)

    # multiply by weight (bf16 op -> round to bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # scalar multiplies, each rounding to bf16 as in the reference
    y = (y.to(tl.float32) * 1.3892).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.0246).to(tl.bfloat16)

    # relu
    y = tl.maximum(y, 0.0).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_scale_relu_kernel[(m,)](
            x, self.rms1_w, out,
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
