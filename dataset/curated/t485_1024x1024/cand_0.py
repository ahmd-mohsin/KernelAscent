import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 485
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_rms_gelu_relu_ln_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_row + offs).to(tl.float32)

    # RMSNorm (computed in fp32, rounded to bf16 exactly where reference rounds)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs).to(tl.float32)
    x = (x * w).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based), rounded to bf16 as in the reference
    x = (0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)

    # ReLU (exact on bf16 values, no extra rounding needed)
    x = tl.maximum(x, 0.0)

    # LayerNorm with fp32 statistics (matches PyTorch bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y_ptr + row * stride_row + offs, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul on tensor cores (bf16), identical to reference
        x = x @ self.W0
        x = x.contiguous()

        M_rows, N = x.shape
        y = torch.empty_like(x)

        _fused_rms_gelu_relu_ln_kernel[(M_rows,)](
            x, self.rms1_w, self.ln4_g, self.ln4_b, y,
            x.stride(0),
            N=N,
            BLOCK=N,
            num_warps=8,
        )
        return y
