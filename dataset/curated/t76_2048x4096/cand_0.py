import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 76
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_rms_gelu_ln_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x16 = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    x = x16.to(tl.float32)

    # ---- RMSNorm (compute in fp32, round to fp16 before weight mul) ----
    ms = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(ms + 1e-6)
    z16 = (x * inv_rms).to(tl.float16)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z16 = (z16.to(tl.float32) * w).to(tl.float16)

    # ---- exact GELU (erf) in fp32, round to fp16 ----
    zf = z16.to(tl.float32)
    g = 0.5 * zf * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # ---- LayerNorm in fp32 ----
    gf = g16.to(tl.float32)
    mean = tl.sum(gf, axis=0) / N
    diff = tl.where(mask, gf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y16 = (diff * rstd * gamma + beta).to(tl.float16)

    # ---- final scalar scale (fp32 opmath, round to fp16) ----
    out = (y16.to(tl.float32) * 1.1664).to(tl.float16)
    tl.store(Out_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # heavy GEMM via cuBLAS (tensor cores)
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_ln_kernel[(Mrows,)](
            y, self.rms1_w, self.ln3_g, self.ln3_b, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
