import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 748
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, G_ptr, B_ptr, B3_ptr, B5_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offs = row * N + cols

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), round to bf16 to match kernel boundary
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm over last dim (fp32 accumulation)
    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b3
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b3
    y = y.to(tl.bfloat16).to(tl.float32)

    # GELU
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b5
    b5 = tl.load(B5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b5

    tl.store(Y_ptr + offs, y.to(tl.bfloat16), mask=mask)


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
        if not x.is_cuda:
            x = x @ self.W0
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = x + self.b3
            x = F.gelu(x)
            x = x + self.b5
            return x

        h = x @ self.W0  # (M, 4096), cuBLAS bf16 tensor-core GEMM
        h = h.contiguous()
        rows, N = h.shape[0], h.shape[1]
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            h, self.ln2_g, self.ln2_b, self.b3, self.b5, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
