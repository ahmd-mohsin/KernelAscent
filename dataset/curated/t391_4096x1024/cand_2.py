import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 391
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _scale_ln_kernel(
    X, G, B, Y,
    N,
    eps,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # match reference: x = x * 1.114 rounded back to fp16 before layer_norm
    x = (x * scale).to(tl.float16).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two GEMMs via cuBLAS tensor cores (kept separate for numerical equivalence)
        x = x @ self.W0
        x = x @ self.W1

        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(N)
        _scale_ln_kernel[(Mrows,)](
            x, self.ln3_g, self.ln3_b, y,
            N, 1e-5, 1.114,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
