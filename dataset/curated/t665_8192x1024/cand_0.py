import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 665
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N, eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X_ptr + row * N + offs

    # ---- LayerNorm (fp32 math, single rounding to bf16, like ATen) ----
    x = tl.load(ptr).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + eps_ln)
    g = tl.load(G_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = d * rstd * g + b
    y_bf = y.to(tl.bfloat16)          # rounding point of F.layer_norm output

    # ---- RMSNorm (explicit .float() in reference) ----
    yf = y_bf.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = yf * tl.math.rsqrt(ms + eps_rms)
    r_bf = r.to(tl.bfloat16)          # .to(x.dtype) rounding point

    # ---- * rms2_w (bf16 op computed in fp32, rounded to bf16) ----
    w = tl.load(W_ptr + offs).to(tl.float32)
    t = (r_bf.to(tl.float32) * w).to(tl.bfloat16)

    # ---- * 1.451 (scalar fp32 math, rounded to bf16) ----
    out = (t.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (same as reference matmul)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_ln_rms_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N, 1e-5, 1e-6, 1.451,
            BLOCK=512,
            num_warps=4,
        )
        return out
