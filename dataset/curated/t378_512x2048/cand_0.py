import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 378
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_rms_relu_ln_kernel(
    X, W0, G2, B2, OUT,
    N, stride_x, stride_o,
    eps_rms, eps_ln,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in float32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps_rms)
    xn = x * inv
    # round to bf16 (matches .to(x.dtype))
    xn = xn.to(tl.bfloat16).to(tl.float32)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    h = xn * w0
    # elementwise bf16*bf16 in PyTorch computes in fp32 then rounds to bf16
    h = h.to(tl.bfloat16).to(tl.float32)

    # ReLU
    h = tl.maximum(h, 0.0)

    # LayerNorm in float32
    mean = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    d = tl.where(mask, h - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)
    y = d * rstd

    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * g + b

    tl.store(OUT + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_rms_relu_ln_kernel[(Mrows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, out,
            N, x2.stride(0), out.stride(0),
            1e-6, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
