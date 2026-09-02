import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 242
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_scale_rms_ln_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (bf16) -> fp32
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.303 (computed in fp32, rounded back to bf16 like PyTorch)
    x = (x * SCALE).to(tl.bfloat16).to(tl.float32)

    # RMSNorm: mean of squares in fp32
    ms = tl.sum(x * x, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + EPS_RMS)
    r = (x * rs).to(tl.bfloat16).to(tl.float32)

    # multiply by rms weight (fp32 opmath, rounded to bf16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (r * w).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch bf16 layer_norm acc type)
    mean = tl.sum(z, axis=0) / N
    d = tl.where(mask, z - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = d * inv * g + b

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = x @ self.W0

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        rows = h2.shape[0]

        y = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_scale_rms_ln_kernel[(rows,)](
            h2, self.rms2_w, self.ln3_g, self.ln3_b, y,
            N, h2.stride(0), y.stride(0),
            SCALE=1.303,
            EPS_RMS=1e-6,
            EPS_LN=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
