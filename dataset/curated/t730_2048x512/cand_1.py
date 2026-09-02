import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 730
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _dual_ln_bias_kernel(
    X, G1, B1, G2, B2, B3, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 accumulation, like PyTorch's bf16 layer_norm)
    mean1 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(xc * xc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd1) * g1 + b1

    # round to bf16 (matches the bf16 intermediate between the two layer_norms)
    y = y.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(y, axis=0) / N
    yc = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (yc * rstd2) * g2 + b2

    # round to bf16, then bf16 add of bias (matches reference: x + b3 in bf16)
    z_bf16 = z.to(tl.bfloat16)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    out = z_bf16 + b3

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            h = x @ self.W0
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            h = F.layer_norm(h, (h.shape[-1],), self.ln2_g, self.ln2_b)
            return h + self.b3

        # GEMM on tensor cores (bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _dual_ln_bias_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, out,
            N=N, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
