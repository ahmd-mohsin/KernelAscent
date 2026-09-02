import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 749
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_rms_gelu_relu_ln_kernel(
    X_ptr, RW_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, then scale by weight in fp16 precision)
    rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    xh = (x * rms).to(tl.float16).to(tl.float32)
    rw = tl.load(RW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xh = (xh * rw).to(tl.float16).to(tl.float32)

    # exact (erf) GELU, opmath fp32 then round to fp16 like PyTorch half kernels
    g = 0.5 * xh * (1.0 + tl.math.erf(xh * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # scale
    g = (g * 1.2045).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 accumulation, eps=1e-5)
    g_masked = tl.where(mask, g, 0.0)
    mean = tl.sum(g_masked, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.rsqrt(var + 1e-5)

    gam = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bet = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * gam + bet
    tl.store(Y_ptr + row * stride + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x @ self.W0
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.gelu(x)
            x = torch.relu(x)
            x = x * 1.2045
            x = F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)
            return x

        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_relu_ln_kernel[(Mrows,)](
            h, self.rms1_w, self.ln5_g, self.ln5_b, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
