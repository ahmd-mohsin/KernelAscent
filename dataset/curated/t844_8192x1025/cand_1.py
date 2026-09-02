import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 844
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _ln_bias_relu_kernel(
    Y_ptr, OUT_ptr,
    G_ptr, B_ptr, B2_ptr, B3_ptr,
    stride_row,
    eps,
    scale,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    y = tl.load(Y_ptr + row * stride_row + cols).to(tl.float32)

    # LayerNorm statistics in fp32 (matches PyTorch mixed-precision LN on bf16)
    mean = tl.sum(y, axis=0) / N
    diff = y - mean
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols).to(tl.float32)
    b = tl.load(B_ptr + cols).to(tl.float32)

    h = diff * rstd * g + b
    # round to bf16 like layer_norm output
    h = h.to(tl.bfloat16)

    # x + b2 (bf16 elementwise: compute in fp32, round to bf16)
    b2 = tl.load(B2_ptr + cols).to(tl.float32)
    h = (h.to(tl.float32) + b2).to(tl.bfloat16)

    # x + b3
    b3 = tl.load(B3_ptr + cols).to(tl.float32)
    h = (h.to(tl.float32) + b3).to(tl.bfloat16)

    # relu (exact on bf16)
    h = tl.maximum(h, 0.0).to(tl.bfloat16)

    # x * scalar (compute in fp32, round back to bf16)
    h = (h.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(OUT_ptr + row * stride_row + cols, h)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (bf16)
        y = x @ self.W0
        y = y.contiguous()

        rows, N = y.shape
        out = torch.empty_like(y)

        _ln_bias_relu_kernel[(rows,)](
            y, out,
            self.ln1_g, self.ln1_b, self.b2, self.b3,
            y.stride(0),
            1e-5,
            1.4218,
            N=N,
            BLOCK=N,
            num_warps=8,
        )
        return out
