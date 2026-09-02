import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 130
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_ln_gelu_bias(
    X_ptr, G_ptr, B_ptr, B5_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul result (bf16), apply scale (rounded to bf16 like the reference)
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = (x * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch internal accumulation), output rounded to bf16
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    p = (num / denom).to(tl.bfloat16).to(tl.float32)
    p = tl.where(mask, p, 0.0)

    # layer norm in fp32, output rounded to bf16
    mean = tl.sum(p, axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU in fp32, output rounded to bf16
    y = (0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))).to(tl.bfloat16)

    # bias add in bf16
    b5 = tl.load(B5_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    out = y + b5

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (identical op to reference)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_softmax_ln_gelu_bias[(Mrows,)](
            h, self.ln3_g, self.ln3_b, self.b5, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.4882,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
