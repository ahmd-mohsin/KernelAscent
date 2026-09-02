import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 453
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_relu_ln_bias_gelu2(
    X, G, B, B3, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    # ReLU
    x = tl.maximum(x, 0.0)

    # LayerNorm (fp32 accumulation, like PyTorch's mixed-precision LN)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # round to bf16 (LN output dtype), like the reference
    y = y.to(tl.bfloat16).to(tl.float32)

    # bias add (bf16 output rounding)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)

    # GELU (exact, erf-based) twice with bf16 rounding between
    SQRT1_2: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQRT1_2))
    y = y.to(tl.bfloat16).to(tl.float32)
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQRT1_2))

    tl.store(Y + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _fused_relu_ln_bias_gelu2[(M_,)](
            h, self.ln2_g, self.ln2_b, self.b3, out,
            N_, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
