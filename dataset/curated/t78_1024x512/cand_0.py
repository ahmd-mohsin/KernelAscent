import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 78
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _double_ln_kernel(
    X, Y, G0, B0, G1, B1,
    D_: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) / D_
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D_
    rstd = 1.0 / tl.sqrt(var + EPS)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g0 + b0

    # round to bf16 to match reference intermediate precision
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / D_
    yc = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / D_
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    z = yc * rstd2 * g1 + b1

    tl.store(Y + row * D_ + cols, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return (x @ self.W2 + self.b3) * 1.3616

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]

        z = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _double_ln_kernel[(n_rows,)](
            x2, z,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b,
            D_=d, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )

        # fused matmul + bias + scale via cuBLAS epilogue
        out = torch.addmm(self.b3, z, self.W2, beta=1.3616, alpha=1.3616)
        return out.view(*orig_shape[:-1], self.W2.shape[1])
