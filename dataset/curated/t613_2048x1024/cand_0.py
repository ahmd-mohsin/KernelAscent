import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 613
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, B1_ptr, RW_ptr, B4_ptr, G_ptr, B_ptr, OUT_ptr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    base = row * N

    # x = x + b1  (bf16 elementwise add computed in fp32, rounded to bf16)
    x = tl.load(X_ptr + base + offs).to(tl.float32)
    b1 = tl.load(B1_ptr + offs).to(tl.float32)
    x = (x + b1).to(tl.bfloat16)

    # RMSNorm: fp32 accumulate, rsqrt, cast to bf16, then * rms2_w
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    x = (xf * r).to(tl.bfloat16)
    rw = tl.load(RW_ptr + offs).to(tl.float32)
    x = (x.to(tl.float32) * rw).to(tl.bfloat16)

    # exact (erf-based) GELU, fp32 opmath, rounded to bf16
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = g.to(tl.bfloat16)

    # x = x + b4
    x = (x.to(tl.float32) + tl.load(B4_ptr + offs).to(tl.float32)).to(tl.bfloat16)

    # LayerNorm in fp32
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = xf - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    gw = tl.load(G_ptr + offs).to(tl.float32)
    bb = tl.load(B_ptr + offs).to(tl.float32)
    y = d * rstd * gw + bb

    tl.store(OUT_ptr + base + offs, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Tensor-core matmul (cuBLAS)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        _fused_post_kernel[(rows,)](
            h, self.b1, self.rms2_w, self.b4, self.ln5_g, self.ln5_b, out,
            N=N,
            num_warps=4,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
