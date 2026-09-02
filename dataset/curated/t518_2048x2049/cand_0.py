import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 518
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, W0, W1, B2, G, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    # ---- load input row (bf16 -> fp32) ----
    xf = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms0 = tl.sum(xf * xf, axis=0) / D
    rs0 = 1.0 / tl.sqrt(ms0 + 1e-6)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    # (xf * rs) rounded to bf16, then bf16-mul (fp32 opmath, round to bf16)
    x1 = (xf * rs0).to(tl.bfloat16).to(tl.float32)
    x1 = (x1 * w0).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 1 ----
    ms1 = tl.sum(x1 * x1, axis=0) / D
    rs1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = (x1 * rs1).to(tl.bfloat16).to(tl.float32)
    x2 = (x2 * w1).to(tl.bfloat16).to(tl.float32)

    # ---- add bias (bf16 add, fp32 opmath) ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    x3 = (x2 + b2).to(tl.bfloat16).to(tl.float32)

    # ---- scale (bf16 mul with fp32 scalar opmath) ----
    x4 = (x3 * 1.1767).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, eps=1e-5) ----
    x4m = tl.where(mask, x4, 0.0)
    mean = tl.sum(x4m, axis=0) / D
    diff = tl.where(mask, x4 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * g + b

    tl.store(Y + row * D + cols, y.to(tl.bfloat16), mask=mask)


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
            return self._forward_ref(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_norm_kernel[(n_rows,)](
            x2d, self.rms0_w, self.rms1_w, self.b2, self.ln4_g, self.ln4_b, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        x = x + self.b2
        x = x * 1.1767
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x
