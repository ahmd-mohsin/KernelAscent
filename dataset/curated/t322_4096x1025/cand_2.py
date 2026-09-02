import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 322
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_relu_ln_rms_kernel(
    x_ptr, g_ptr, b_ptr, w_ptr, out_ptr,
    N,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # LayerNorm (fp32 accumulation, like PyTorch bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.math.rsqrt(var + eps_ln)

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * g + b

    # round to bf16 (layer_norm output dtype)
    y = y.to(tl.bfloat16)

    # x * 1.4215 in bf16 (fp32 compute, bf16 round)
    y = (y.to(tl.float32) * scale).to(tl.bfloat16)

    # RMSNorm in fp32
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + eps_rms)
    z = (yf * r).to(tl.bfloat16)

    # multiply by rms weight (bf16 elementwise -> fp32 compute, bf16 round)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (z.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(out_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x * 1.4215
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_ln_rms_kernel[(rows,)](
            x2, self.ln1_g, self.ln1_b, self.rms3_w, out,
            N,
            1e-5, 1e-6, 1.4215,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
