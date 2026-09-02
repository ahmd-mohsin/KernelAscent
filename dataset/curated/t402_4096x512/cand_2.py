import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 402
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, LN2G, LN2B, RMSW, LN5G, LN5B, OUT_ptr,
    N, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then cast back to bf16 like PyTorch
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (fp32 internal, bf16 output)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    xn = xc * tl.math.rsqrt(var + 1e-5)
    g = tl.load(LN2G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN2B + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xn * g + b).to(tl.bfloat16).to(tl.float32)

    # scale by 1.003 (fp32 opmath, bf16 result)
    x = (x * 1.003).to(tl.bfloat16).to(tl.float32)

    # RMSNorm (fp32 compute, cast to bf16, then bf16*bf16 weight mult in fp32 opmath)
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(RMSW + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2 (fp32 internal, bf16 output)
    mean2 = tl.sum(x, axis=0) / N
    xc2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(xc2 * xc2, axis=0) / N
    xn2 = xc2 * tl.math.rsqrt(var2 + 1e-5)
    g5 = tl.load(LN5G + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(LN5B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn2 * g5 + b5).to(tl.bfloat16)

    tl.store(OUT_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # tensor-core bf16 GEMM
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_post_kernel[(m,)](
            h, self.ln2_g, self.ln2_b, self.rms4_w, self.ln5_g, self.ln5_b, out,
            n, BLOCK=BLOCK,
            num_warps=16,
        )
        return out
