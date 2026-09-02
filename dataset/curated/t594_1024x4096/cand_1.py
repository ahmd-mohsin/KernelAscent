import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 594
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_bias_relu_bias_ln(
    Y_ptr, B1_ptr, B3_ptr, G_ptr, B_ptr, OUT_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Load GEMM output row and biases (keep elementwise math in fp16 to match reference)
    y = tl.load(Y_ptr + row * N + cols, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0)

    z16 = tl.maximum(y + b1, 0.0) + b3          # fp16 arithmetic (matches reference)
    z = z16.to(tl.float32)                      # LN computed in fp32 (matches F.layer_norm)

    mean = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    out = (z - mean) * rstd * g + b
    tl.store(OUT_ptr + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        y = x @ self.W0

        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_relu_bias_ln[(m,)](
            y, self.b1, self.b3, self.ln4_g, self.ln4_b, out,
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
