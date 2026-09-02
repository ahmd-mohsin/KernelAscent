import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 153
M, D, DT = 4096, 513, torch.bfloat16


@triton.jit
def _fused_rms_softmax_ln_kernel(
    X, W, G, B, OUT,
    N, stride_x, stride_o,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (computed in fp32, cast to bf16, then bf16*bf16 weight with fp32 opmath) ----
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    t = (x * r).to(tl.bfloat16)  # round like .to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (t.to(tl.float32) * w).to(tl.bfloat16)  # bf16 result of elementwise mul

    # ---- Softmax (fp32 accumulate, bf16 output) ----
    yf = y.to(tl.float32)
    yf_m = tl.where(mask, yf, float('-inf'))
    mx = tl.max(yf_m, axis=0)
    e = tl.exp(yf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16)  # softmax output rounded to bf16

    # ---- LayerNorm (fp32 math on bf16 input, bf16 output) ----
    pf = p.to(tl.float32)
    pf = tl.where(mask, pf, 0.0)
    mean = tl.sum(pf, axis=0) / N
    diff = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = ((pf - mean) * inv * g + b).to(tl.bfloat16)  # layer_norm output in bf16

    # ---- final scale (bf16 * scalar, fp32 opmath then round) ----
    out = (ln.to(tl.float32) * SCALE).to(tl.bfloat16)
    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x * 1.3856

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_ln_kernel[(rows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, out,
            N, x2.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5, SCALE=1.3856,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
