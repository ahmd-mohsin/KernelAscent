import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 784
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_relu_bias_ln_ln(
    X, B2, G3, B3, G4, B4, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)

    # relu + bias (bf16 add semantics: fp32 add, round to bf16)
    x = tl.maximum(x, 0.0)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (fp32 compute)
    n = N.to(tl.float32) if hasattr(N, "to") else N
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x - mean) * rstd * g3 + b3
    # round to bf16 between the two layernorms (matches reference)
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2 (fp32 compute)
    mean2 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(diff2 * diff2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean2) * rstd2 * g4 + b4

    tl.store(Y + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores (same as reference)
        h = torch.matmul(x, self.W0)

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_bias_ln_ln[(rows,)](
            h2, self.b2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
