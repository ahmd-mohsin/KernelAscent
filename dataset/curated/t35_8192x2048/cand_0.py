import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 35
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln_rms_relu_kernel(
    X, G, B, W, Y,
    N,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- LayerNorm (fp32 internals, like PyTorch) ----
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    # cast back to bf16 (layer_norm output dtype), then re-read as fp32 (matches _xf = x.float())
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    # ---- RMSNorm in fp32 ----
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + EPS_RMS)
    z = (yf * r).to(tl.bfloat16)

    # ---- scale by rms2_w in bf16 (matches bf16 elementwise mul), then ReLU ----
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = z * w
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already near-optimal on A100 tensor cores)
        h = x @ self.W0

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]

        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_rms_relu_kernel[(rows,)](
            h2, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
