import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 851
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _relu_ln_relu_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # relu (pre-LN)
    x = tl.maximum(x, 0.0)

    # layer norm statistics in fp32 (matches PyTorch bf16 LN internals)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    y = y.to(tl.bfloat16)
    # relu (post-LN), applied on bf16 exactly like the reference
    y = tl.maximum(y, y - y)

    tl.store(Y_ptr + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 via cuBLAS (tensor cores)
        h = x @ self.W0  # (M, 1024), bf16

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _relu_ln_relu_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2 via cuBLAS, then scale (in-place to avoid extra allocation)
        y = out @ self.W4
        y.mul_(1.2434)
        return y
