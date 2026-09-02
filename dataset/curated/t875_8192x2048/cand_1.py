import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 875
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln_gelu_kernel(
    X, G, B, B3, Y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, as PyTorch does for bf16 inputs)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # round to bf16 (layer_norm output dtype in reference)
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact (erf) GELU, rounded to bf16 like F.gelu output
    ge = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    ge = ge.to(tl.bfloat16).to(tl.float32)

    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    # x + b3, rounded to bf16, then scale
    out = (ge + b3).to(tl.bfloat16).to(tl.float32) * scale

    tl.store(Y + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_gelu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.b3, out,
            N, 1e-5, 1.3811,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
