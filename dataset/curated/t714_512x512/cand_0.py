import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 714
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_bias_relu_ln_rms_kernel(
    X_ptr, Bias_ptr, G_ptr, Beta_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul output (fp16) and bias (fp16); add in fp16 like PyTorch
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(Bias_ptr + cols, mask=mask, other=0.0)
    x = x + b
    # relu (exact regardless of precision)
    x = tl.where(x > 0, x, x * 0)

    # LayerNorm in fp32 (matches PyTorch half layernorm internal fp32)
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + beta
    yh = y.to(tl.float16)  # cast to fp16 (LN output dtype)

    # RMSNorm: recompute from fp16 value cast to fp32 (matches reference)
    yf = yh.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    out16 = (yf * r).to(tl.float16)

    # multiply by rms weight in fp16 (matches half * half in PyTorch)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out16 = out16 * w
    tl.store(Y_ptr + row * stride_y + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.reshape(-1, N)

        # matmul via cuBLAS tensor cores
        h = torch.matmul(x2, self.W0)
        h = h.contiguous()

        rows = h.shape[0]
        Nout = h.shape[1]
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(Nout)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_bias_relu_ln_rms_kernel[(rows,)](
            h, self.b1, self.ln3_g, self.ln3_b, self.rms4_w, out,
            Nout, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.reshape(*orig_shape[:-1], Nout)
