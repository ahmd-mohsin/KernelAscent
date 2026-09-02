import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 233
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_ln_bias_gelu_kernel(
    X_ptr, G_ptr, B_ptr, B2_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm statistics in fp32 (matches PyTorch's internal fp32 accumulation)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    # layer_norm output is rounded to bf16 before the next op
    y_bf = y.to(tl.bfloat16).to(tl.float32)

    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # bf16 add semantics: compute in fp32, round result to bf16
    s = (y_bf + b2).to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * s * (1.0 + tl.math.erf(s * INV_SQRT2))

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (bit-identical to reference matmul)
        h = x @ self.W0

        if not h.is_cuda:
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            h = h + self.b2
            return F.gelu(h)

        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK_N = triton.next_power_of_2(N)
        _fused_ln_bias_gelu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.b2, out,
            N, h.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
