import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 518
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_rms_rms_bias_scale_ln(
    X, W0, W1, B2, G, B, Y,
    D, stride_x, stride_y,
    eps_rms, eps_ln, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 (fp32 stats, round to bf16, bf16*bf16 with fp32 opmath) ----
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + eps_rms)
    y = (x * r).to(tl.bfloat16).to(tl.float32)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (y * w0).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 1 ----
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + eps_rms)
    y = (x * r).to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (y * w1).to(tl.bfloat16).to(tl.float32)

    # ---- bias add (bf16 result, fp32 opmath) ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    # ---- scalar scale (bf16 result, fp32 opmath) ----
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, bf16 output) ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps_ln)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = d * rstd * g + b

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._ref_forward(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_rms_rms_bias_scale_ln[(n_rows,)](
            x2, self.rms0_w, self.rms1_w, self.b2, self.ln4_g, self.ln4_b, y,
            d, x2.stride(0), y.stride(0),
            1e-6, 1e-5, 1.1767,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

    def _ref_forward(self, x):
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        x = x + self.b2
        x = x * 1.1767
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x
