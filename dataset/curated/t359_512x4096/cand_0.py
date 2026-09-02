import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 359
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_ln_rms_ln_kernel(
    X, G0, B0, W2, G3, B3, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # --- LayerNorm 0 (fp32 math, bf16 rounding at output like PyTorch) ---
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xc * inv * g0 + b0).to(tl.bfloat16).to(tl.float32)

    # --- scale by 1.0166 (bf16 rounding) ---
    x = (x * 1.0166).to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm (fp32 math from bf16 input) ---
    x = tl.where(mask, x, 0.0)
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # --- LayerNorm 3 ---
    mean2 = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    xc2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(xc2 * xc2, axis=0) / D
    inv2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xc2 * inv2 * g3 + b3).to(tl.bfloat16).to(tl.float32)

    # --- scale by 1.0289 ---
    y = (x * 1.0289).to(tl.bfloat16)

    tl.store(Y + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = x * 1.0166
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = x * 1.0289
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_rms_ln_kernel[(n_rows,)](
            x2, self.ln0_g, self.ln0_b, self.rms2_w, self.ln3_g, self.ln3_b, y,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
