import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 245
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_ln_bias_rms_kernel(
    x_ptr, g_ptr, b_ptr, b1_ptr, w_ptr, out_ptr,
    D: tl.constexpr,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (computed in fp32, output rounded to bf16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    inv = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * inv * g + b
    y = y.to(tl.bfloat16)  # matches bf16 output of F.layer_norm

    # ---- bias add (bf16 tensors, opmath fp32, single rounding) ----
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) + b1).to(tl.bfloat16)

    # ---- RMSNorm (fp32 math from bf16 input, round to bf16) ----
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / D
    r = yf * (1.0 / tl.sqrt(ms + eps_rms))
    r = r.to(tl.bfloat16)  # .to(x.dtype)

    # ---- multiply by rms2_w (bf16 * bf16 -> fp32 math -> bf16) ----
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = (r.to(tl.float32) * w).to(tl.bfloat16)

    # ---- scalar scale (fp32 math -> bf16) ----
    out = (t.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: original path
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y + self.b1
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms2_w
            return y * 1.4116

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        out = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_bias_rms_kernel[(m,)](
            xc, self.ln0_g, self.ln0_b, self.b1, self.rms2_w, out,
            d, 1e-5, 1e-6, 1.4116,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
