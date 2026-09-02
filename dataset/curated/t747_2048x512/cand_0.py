import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 747
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G0, B0, B1, W2, OUT,
    stride_xm, stride_om,
    N, LN_EPS, RMS_EPS, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, matches PyTorch's internal fp32 for bf16) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g0 + b0
    # cast to bf16 (layer_norm output dtype)
    y = y.to(tl.bfloat16)

    # ---- add b1 (bf16 op, fp32 opmath) ----
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) + b1).to(tl.bfloat16)

    # ---- RMSNorm in fp32 ----
    xf = y.to(tl.float32)
    xf_m = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf_m * xf_m, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    z = (xf * r).to(tl.bfloat16)

    # ---- multiply rms2_w (bf16, fp32 opmath) ----
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) * w2).to(tl.bfloat16)

    # ---- scalar scale ----
    z = (z.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(OUT + row * stride_om + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y + self.b1
            _xf = y.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms2_w
            return y * 1.1319

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(Mrows,)](
            x2, self.ln0_g, self.ln0_b, self.b1, self.rms2_w, out,
            x2.stride(0), out.stride(0),
            N, 1e-5, 1e-6, 1.1319,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
