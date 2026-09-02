import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 583
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _fused_gelu_ln_relu_rms_kernel(
    X, G, B, W, Y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then cast back to fp16 like PyTorch
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # LayerNorm (stats in fp32)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (g - mean) * inv_std * gamma + beta
    ln = ln.to(tl.float16)

    # ReLU (fp16)
    r = tl.maximum(ln, ln * 0)

    # RMSNorm in fp32, cast to fp16, then multiply by fp16 weight
    rf = r.to(tl.float32)
    ms = tl.sum(tl.where(mask, rf * rf, 0.0), axis=0) / N
    rr = rf * tl.math.rsqrt(ms + 1e-6)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = rr.to(tl.float16) * w

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        m, n = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_ln_relu_rms_kernel[(m,)](
            h, self.ln2_g, self.ln2_b, self.rms4_w, y,
            n,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
