import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 307
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln_softmax_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N, stride,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X_ptr + row * stride + offs

    # ---- load row in fp32 (opmath, matches PyTorch bf16 handling) ----
    x = tl.load(ptr).to(tl.float32)

    # ---- LayerNorm (fp32 math, biased variance, eps=1e-5) ----
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)
    g = tl.load(G_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = xc * rstd * g + b
    # round to bf16 as PyTorch does between ops
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 math, output rounded to bf16) ----
    mx = tl.max(y, axis=0)
    e = tl.exp(y - mx)
    s = tl.sum(e, axis=0)
    y = e / s
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- scale (bf16 op with fp32 opmath) + relu ----
    y = y * scale
    y = y.to(tl.bfloat16).to(tl.float32)
    y = tl.maximum(y, 0.0)

    # ---- RMSNorm in fp32, cast bf16, then * weight (fp32 opmath) ----
    ms = tl.sum(y * y, axis=0) / N
    y = y * (1.0 / tl.sqrt(ms + eps_rms))
    y = y.to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + offs).to(tl.float32)
    y = y * w

    tl.store(Y_ptr + row * stride + offs, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already optimal on A100 tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_ln_softmax_rms_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms5_w, out,
            N, h.stride(0),
            1e-5, 1e-6, 1.3367,
            BLOCK=512,
            num_warps=4,
        )
        return out
