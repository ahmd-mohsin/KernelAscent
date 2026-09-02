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
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # LayerNorm (fp32 accumulation, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    # cast to bf16 (layer_norm output dtype), then add b3 like the eager path
    y = y.to(tl.bfloat16)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    y = y + b3

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # first GELU (exact erf, computed in fp32 like PyTorch opmath)
    yf = y.to(tl.float32)
    yf = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    # round to bf16 between the two GELUs (matches eager intermediate)
    yf = yf.to(tl.bfloat16).to(tl.float32)

    # second GELU
    yf = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))

    tl.store(Y + row * N + cols, yf.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        m, n = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        _fused_relu_ln_bias_gelu2[(m,)](
            h, self.ln2_g, self.ln2_b, self.b3, out,
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
