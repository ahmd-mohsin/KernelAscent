import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 856
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_gelu_ln_bias(
    X_ptr, Y_ptr, G_ptr, B_ptr, B3_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU, computed in fp32 then rounded to fp16 to match reference
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # layernorm stats in fp32 (matches PyTorch fp16 layer_norm internals)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = (diff * rstd * gamma + beta).to(tl.float16)

    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0)
    y = y + b3

    tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = x @ self.W0

        M_rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_ln_bias[(M_rows,)](
            h, out, self.ln2_g, self.ln2_b, self.b3,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
