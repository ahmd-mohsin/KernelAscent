import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 927
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_ln_ln_scale_rms_kernel(
    X, G1, B1, G2, B2, RW, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X + row * stride_x + cols).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, output rounded to fp16 like eager) ----
    mean1 = tl.sum(x, axis=0) / N
    xc = x - mean1
    var1 = tl.sum(xc * xc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g1 = tl.load(G1 + cols).to(tl.float32)
    b1 = tl.load(B1 + cols).to(tl.float32)
    y = (xc * rstd1 * g1 + b1).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(y, axis=0) / N
    yc = y - mean2
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g2 = tl.load(G2 + cols).to(tl.float32)
    b2 = tl.load(B2 + cols).to(tl.float32)
    z = (yc * rstd2 * g2 + b2).to(tl.float16).to(tl.float32)

    # ---- scalar mul (opmath fp32, round to fp16) ----
    z = (z * 1.1137).to(tl.float16).to(tl.float32)

    # ---- RMSNorm in fp32, cast to fp16, then elementwise mul by weight ----
    ms = tl.sum(z * z, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    zn = (z * r).to(tl.float16).to(tl.float32)
    rw = tl.load(RW + cols).to(tl.float32)
    out = (zn * rw).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0

        orig_shape = h.shape
        n = orig_shape[-1]
        h2 = h.reshape(-1, n)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        rows = h2.shape[0]

        out = torch.empty_like(h2)
        _fused_ln_ln_scale_rms_kernel[(rows,)](
            h2, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.rms4_w, out,
            h2.stride(0), out.stride(0),
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out.reshape(orig_shape)
