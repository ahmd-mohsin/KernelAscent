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
    X, OUT,
    G1, B1, G2, B2,
    N, EPS,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU 1 (exact, erf) -- compute in fp32, round to fp16 like PyTorch
    inv_sqrt2 = 0.7071067811865476
    y = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    y = y.to(tl.float16).to(tl.float32)

    # GELU 2
    y = y * 0.5 * (1.0 + tl.math.erf(y * inv_sqrt2))
    y = y.to(tl.float16).to(tl.float32)

    # LayerNorm 1 (fp32 stats)
    mean1 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d1 = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d1 * rstd1 * g1 + b1
    z = z.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g2 + b2

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


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
        y = x @ self.W0  # cuBLAS tensor-core GEMM
        if not y.is_cuda:
            y = F.gelu(y)
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)
            return y

        orig_shape = y.shape
        N = orig_shape[-1]
        y2 = y.contiguous().view(-1, N)
        Mrows = y2.shape[0]
        out = torch.empty_like(y2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 512 else 4
        _fused_gelu2_ln2_kernel[(Mrows,)](
            y2, out,
            self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b,
            N, 1e-5,
            y2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
