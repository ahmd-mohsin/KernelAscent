import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 704
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_relu_ln_gelu_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N, eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # LayerNorm (fp32 math, biased variance) -> cast to bf16 like PyTorch
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # Exact GELU (erf), fp32 math then round to bf16 (matches PyTorch opmath)
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32, round to bf16
    yz = tl.where(mask, y, 0.0)
    ms = tl.sum(yz * yz, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps_rms)
    y = (y * inv).to(tl.bfloat16).to(tl.float32)

    # * rms4_w (bf16*bf16 done in fp32 opmath, rounded back)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # scalar scale (fp32 opmath, rounded back)
    y = y * scale

    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_ln_gelu_rms_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.rms4_w, out,
            N, 1e-5, 1e-6, 1.1698,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
