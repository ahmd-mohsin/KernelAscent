import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 974
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_rms_ln_kernel(
    X_ptr, W1_ptr, G_ptr, B_ptr, Y_ptr,
    N, eps_rms, eps_ln, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, rounded to bf16 like reference)
    ms = tl.sum(x * x, axis=0) / N
    r = x * tl.math.rsqrt(ms + eps_rms)
    r = r.to(tl.bfloat16).to(tl.float32)

    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = (r * w1).to(tl.bfloat16).to(tl.float32)  # bf16 elementwise mul rounding

    # LayerNorm (fp32 internal math, matching F.layer_norm on bf16 input)
    mean = tl.sum(tl.where(mask, t, 0.0), axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + eps_ln)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b
    y = y.to(tl.bfloat16).to(tl.float32)  # layer_norm output rounded to bf16

    y = y * scale  # scalar mul then round to bf16
    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape[0], x.shape[-1]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_kernel[(rows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            N, 1e-6, 1e-5, 1.3244,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
