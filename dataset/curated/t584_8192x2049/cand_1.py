import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 584
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_gelu_double_rms_kernel(
    X_ptr, W1_ptr, W2_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32 (matches PyTorch's opmath for bf16), rounded back to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm 1
    ms1 = tl.sum(g * g, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + EPS)
    h = (g * r1).to(tl.bfloat16)  # round normed value to bf16 (matches .to(x.dtype))
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    h = (h.to(tl.float32) * w1).to(tl.bfloat16)  # bf16 * bf16 -> bf16

    # RMSNorm 2
    hf = h.to(tl.float32)
    ms2 = tl.sum(hf * hf, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + EPS)
    h2 = (hf * r2).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    h2 = (h2.to(tl.float32) * w2).to(tl.bfloat16)

    # final scale (bf16 tensor * python float -> fp32 opmath, round to bf16)
    out = (h2.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # fallback reference path
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x * 1.3399

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_double_rms_kernel[(Mrows,)](
            x2d, self.rms1_w, self.rms2_w, y,
            N, x2d.stride(0), y.stride(0),
            EPS=1e-6, SCALE=1.3399,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
