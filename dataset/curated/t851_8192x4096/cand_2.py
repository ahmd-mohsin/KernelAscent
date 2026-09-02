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
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)

    # layernorm (fp32 accumulation, matches PyTorch opmath)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # relu (applying before or after bf16 cast is identical for max(.,0))
    y = tl.maximum(y, 0.0)

    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (tensor cores)
        h = torch.matmul(x, self.W0)  # (M, 1024), bf16

        Mrows, N = h.shape
        y = torch.empty_like(h)
        _relu_ln_relu_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, y,
            N=N, EPS=1e-5, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )

        # GEMM 2 + scale
        out = torch.matmul(y, self.W4)
        out.mul_(1.2434)
        return out
