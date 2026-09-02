import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 858
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_ln_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Load matmul output (bf16) -> fp32
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # Exact (erf) GELU, computed in fp32, rounded to bf16 (matching PyTorch opmath)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm (fp32 accumulation, as in reference: _xf = x.float())
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / N
    r = g * tl.math.rsqrt(ms + 1e-6)
    # cast normalized result back to bf16 (matches .to(x.dtype)), then bf16 * bf16 weight
    r = r.to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    t = (r * w).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch bf16 layer_norm)
    mean = tl.sum(tl.where(mask, t, 0.0), axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)

    gg = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * gg + bb

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Tensor-core matmul (cuBLAS bf16)
        h = x @ self.W0  # (M, 1024) bf16

        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_rms_ln_kernel[(Mrows,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, y,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
