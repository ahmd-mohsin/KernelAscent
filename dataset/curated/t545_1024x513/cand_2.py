import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 545
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_ln_softmax_rms_kernel(
    X_ptr, LN_G_ptr, LN_B_ptr, B3_ptr, RMS_W_ptr, Y_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * N + offs).to(tl.float32)

    # ---- LayerNorm (fp32 math, cast to fp16 like F.layer_norm on half) ----
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN_G_ptr + offs).to(tl.float32)
    b = tl.load(LN_B_ptr + offs).to(tl.float32)
    y = (xc * inv_std * g + b).to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 math, output cast to fp16) ----
    row_max = tl.max(y, axis=0)
    e = tl.exp(y - row_max)
    denom = tl.sum(e, axis=0)
    y = (e / denom).to(tl.float16).to(tl.float32)

    # ---- + b3 (half op -> fp32 opmath, round to fp16) ----
    b3 = tl.load(B3_ptr + offs).to(tl.float32)
    y = (y + b3).to(tl.float16).to(tl.float32)

    # ---- * 1.4014 (half op -> fp32 opmath, round to fp16) ----
    y = (y * 1.4014).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (explicit fp32 in reference, cast to fp16, then * w) ----
    ms = tl.sum(y * y, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (y * r).to(tl.float16).to(tl.float32)
    w = tl.load(RMS_W_ptr + offs).to(tl.float32)
    out = (y * w).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        _fused_ln_softmax_rms_kernel[(M_,)](
            h, self.ln1_g, self.ln1_b, self.b3, self.rms5_w, out,
            N=N_, BLOCK=triton.next_power_of_2(N_),
            num_warps=4,
        )
        return out
