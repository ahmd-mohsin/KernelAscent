import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 610
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _ln_bias_gelu_relu_kernel(
    X_ptr, G_ptr, B_ptr, B2_ptr, Y_ptr,
    N, stride_x, stride_y, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch bf16 layer_norm internals)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # round to bf16 (layer_norm output dtype)
    y = y.to(tl.bfloat16).to(tl.float32)

    # bias add (fp32 opmath, bf16 output rounding)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = y + b2
    z = z.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf) in fp32, round to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))
    gel = gel.to(tl.bfloat16).to(tl.float32)

    # ReLU
    out = tl.maximum(gel, 0.0)

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _ln_bias_gelu_relu_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.b2, out,
            N, h.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
