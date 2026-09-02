import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 47
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_bias_ln_rms(
    X_ptr, B1_ptr, G_ptr, B_ptr, RW_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul result row and bias; replicate bf16 add rounding
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch's fp32 internal compute for bf16)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + EPS_LN)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (d * rstd * g + bb).to(tl.bfloat16).to(tl.float32)

    # scale by 1.0003 with bf16 rounding (matches bf16 tensor * python float)
    z = (z * 1.0003).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 over the bf16 values, then bf16 rounding, then bf16 weight mul
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + EPS_RMS)
    zn = (z * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(RW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (zn * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        # cuBLAS bf16 matmul (tensor cores)
        h = torch.matmul(x2, self.W0)
        h = h.contiguous()
        M_rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_bias_ln_rms[(M_rows,)](
            h, self.b1, self.ln2_g, self.ln2_b, self.rms4_w, y,
            N, h.stride(0), y.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.reshape(*orig_shape[:-1], N)
