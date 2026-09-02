import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 703
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W1_ptr, G_ptr, B_ptr, W5_ptr, Y_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # ---- load row (fp16 -> fp32) ----
    x = tl.load(X_ptr + base + offs).to(tl.float32)

    # ---- RMSNorm 1 ----
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w1 = tl.load(W1_ptr + offs)  # fp16
    # cast normalized value to fp16 first (matches .to(x.dtype)), then * w (fp16)
    x1 = ((x * r).to(tl.float16) * w1).to(tl.float32)

    # ---- LayerNorm (stats + affine in fp32, output fp16) ----
    mean = tl.sum(x1, axis=0) / N
    d = x1 - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = (d * rstd * g + b).to(tl.float16)

    # ---- ReLU + scale (opmath fp32, result fp16) ----
    z = tl.maximum(y.to(tl.float32), 0.0) * 1.455
    zh = z.to(tl.float16)

    # ---- RMSNorm 2 ----
    zf = zh.to(tl.float32)
    ms2 = tl.sum(zf * zf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w5 = tl.load(W5_ptr + offs)  # fp16
    out = (zf * r2).to(tl.float16) * w5

    tl.store(Y_ptr + base + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_post_kernel[(m,)](
            h, self.rms1_w, self.ln2_g, self.ln2_b, self.rms5_w, out,
            N=n, BLOCK=n,
            num_warps=4,
        )
        return out
