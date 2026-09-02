import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 748
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_gelu_ln_gelu_kernel(
    X_ptr, G_ptr, B_ptr, B3_ptr, B5_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), computed in fp32 then rounded back to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.bfloat16).to(tl.float32)

    # LayerNorm with fp32 statistics (matches PyTorch bf16 layer_norm)
    mean = tl.sum(tl.where(mask, g1, 0.0), axis=0) / N
    d = tl.where(mask, g1 - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv_std * gamma + beta
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b3 (fp32 compute, round to bf16 like PyTorch elementwise add)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)

    # GELU again
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b5
    b5 = tl.load(B5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y + b5).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS GEMM (bf16)
        h = h.contiguous()
        M_, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_gelu_kernel[(M_,)](
            h, self.ln2_g, self.ln2_b, self.b3, self.b5, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
