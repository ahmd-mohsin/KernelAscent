import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 528
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_ln_relu_rms_kernel(
    X, G, B, B3, W, Y,
    D: tl.constexpr,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch bf16 layer_norm)
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # layer_norm outputs bf16
    y = y.to(tl.bfloat16).to(tl.float32)

    # x * 1.3076 (fp32 opmath, rounded back to bf16)
    y = (y * scale).to(tl.bfloat16).to(tl.float32)

    # relu (exact)
    y = tl.maximum(y, 0.0)

    # x + b3 (fp32 opmath, rounded back to bf16)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)
    y = tl.where(mask, y, 0.0)

    # RMSNorm in fp32
    ms = tl.sum(y * y, axis=0) / D
    r = 1.0 / tl.sqrt(ms + eps_rms)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    # (_xf * rsqrt).to(bf16) * w  -> bf16 mul executed in fp32 opmath, rounded to bf16
    out = ((y * r).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y + row * D + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y * 1.3076
            y = torch.relu(y)
            y = y + self.b3
            _xf = y.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms4_w
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_relu_rms_kernel[(n_rows,)](
            x2, self.ln0_g, self.ln0_b, self.b3, self.rms4_w, out,
            d, 1e-5, 1e-6, 1.3076,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
