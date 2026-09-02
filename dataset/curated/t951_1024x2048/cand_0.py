import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 951
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_ln_ln_add_kernel(
    X, G1, B1, G2, B2, B3, Y,
    D: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU, computed in fp32 then rounded to bf16 (matches PyTorch op boundary)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (fp32 accumulation, bf16 output)
    mean1 = tl.sum(tl.where(mask, g, 0.0), axis=0) / D
    d1 = tl.where(mask, g - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / D
    inv1 = 1.0 / tl.sqrt(var1 + eps)
    w1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    bb1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d1 * inv1 * w1 + bb1
    y = y.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2 (fp32 accumulation, bf16 output)
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / D
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / D
    inv2 = 1.0 / tl.sqrt(var2 + eps)
    w2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    bb2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    z = d2 * inv2 * w2 + bb2
    z = z.to(tl.bfloat16).to(tl.float32)

    # bias add (fp32 opmath, bf16 output)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (z + b3).to(tl.bfloat16)
    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return y + self.b3

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_ln_ln_add_kernel[(m,)](
            x2, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, out,
            D=d, eps=1e-5, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
