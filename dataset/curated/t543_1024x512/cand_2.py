import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 543
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_gelu2_ln2_kernel(
    X, G3, B3, G4, B4, Y,
    D: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf) #1 -- round to fp16 to match PyTorch elementwise op output dtype
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # GELU #2
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm #1 (fp32 accumulation like PyTorch)
    mean1 = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    xm = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(xm * xm, axis=0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd1 * g3 + b3
    y = y.to(tl.float16).to(tl.float32)  # round like PyTorch output of LN

    # LayerNorm #2
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / D
    ym = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(ym * ym, axis=0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = ym * rstd2 * g4 + b4

    tl.store(Y + row * D + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0

        orig_shape = h.shape
        d = orig_shape[-1]
        h2 = h.contiguous().view(-1, d)
        rows = h2.shape[0]

        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_gelu2_ln2_kernel[(rows,)](
            h2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            d, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
