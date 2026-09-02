import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 809
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_act_ln_kernel(
    X_ptr, Y_ptr,
    G4_ptr, B4_ptr, G5_ptr, B5_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # gelu (exact, erf) computed in fp32, rounded back to bf16 as PyTorch does per-op
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # second gelu
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm 4 (stats in fp32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g4 = tl.load(G4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g4 + b4
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm 5
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g5 = tl.load(G5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g5 + b5

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_act_ln_kernel[(rows,)](
            h, y,
            self.ln4_g, self.ln4_b, self.ln5_g, self.ln5_b,
            N, h.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
