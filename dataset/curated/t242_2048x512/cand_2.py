import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 242
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, OUT_ptr,
    N, stride_x, stride_o,
    eps_rms, eps_ln, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)  # bf16

    # x = x * 1.303 (computed in fp32, rounded back to bf16 like PyTorch)
    xf = x.to(tl.float32) * scale
    xb = xf.to(tl.bfloat16)

    # RMSNorm in fp32
    _xf = xb.to(tl.float32)
    ms = tl.sum(_xf * _xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + eps_rms)
    normed = (_xf * rs).to(tl.bfloat16)

    # multiply by rms weight (bf16 elementwise, fp32 compute rounded to bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = (normed.to(tl.float32) * w).to(tl.bfloat16)

    # LayerNorm in fp32
    rf = r.to(tl.float32)
    mean = tl.sum(rf, axis=0) / N
    diff = tl.where(mask, rf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps_ln)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * g + b

    tl.store(OUT_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_norm_kernel[(m,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, out,
            n, h.stride(0), out.stride(0),
            1e-6, 1e-5, 1.303,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
