import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 801
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load row (bf16 -> fp32)
    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, like PyTorch's bf16 layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    inv_std = tl.math.rsqrt(var + eps_ln)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * inv_std * g + b

    # round to bf16 (layer_norm output dtype), then upcast (matches _xf = x.float())
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    # ---- RMSNorm on the bf16-rounded layernorm output ----
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + eps_rms)

    # (xf * rsqrt).to(bf16)  then  * rms2_w  (fp32 opmath, bf16 result)
    z_bf = (yf * r).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z_bf.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0
        if not h.is_contiguous():
            h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_rms_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N, 1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
