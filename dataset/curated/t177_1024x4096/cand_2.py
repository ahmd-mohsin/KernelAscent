import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 177
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_rms_ln_softmax(
    X, W_RMS, G, B, OUT,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16), apply relu
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)

    # ---- RMSNorm (computed in fp32, cast back to fp16, then * weight) ----
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    x16 = (xf * rrms).to(tl.float16)

    w = tl.load(W_RMS + offs, mask=mask, other=0.0).to(tl.float32)
    v = (x16.to(tl.float32) * w).to(tl.float16)  # fp16 rounding as in reference

    # ---- LayerNorm (fp32 internals, fp16 output) ----
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, 0.0)
    mean = tl.sum(vf, axis=0) / N
    diff = tl.where(mask, vf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd * g + b).to(tl.float16)

    # ---- ReLU ----
    y = tl.maximum(y, 0.0)

    # ---- Softmax (fp32 internals, fp16 output) ----
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    mx = tl.max(yf, axis=0)
    e = tl.exp(yf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_softmax[(Mrows,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
