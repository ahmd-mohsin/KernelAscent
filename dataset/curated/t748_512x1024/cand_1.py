import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 748
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_gelu_ln_bias_gelu_bias(
    X, G, B, B3, B5, Out,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based) in fp32, then round to bf16 like PyTorch's
    # intermediate tensor storage
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation, biased variance, eps=1e-5)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b3
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b3
    y = y.to(tl.bfloat16).to(tl.float32)

    # GELU
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b5
    b5 = tl.load(B5 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b5

    tl.store(Out + row * N + cols, y.to(tl.bfloat16), mask=mask)


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
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]

        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_bias_gelu_bias[(rows,)](
            h2, self.ln2_g, self.ln2_b, self.b3, self.b5, out,
            N,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out.view(orig_shape)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
