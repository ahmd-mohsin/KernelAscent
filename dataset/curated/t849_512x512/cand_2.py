import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 849
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, g_ptr, b_ptr, w_ptr, out_ptr,
    N, stride_row,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0 (bf16 add: fp32 compute, round to bf16)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * inv_std) * g + bb
    y = y.to(tl.bfloat16).to(tl.float32)  # round to bf16 (layer_norm output dtype)

    # RMSNorm: fp32 compute, round to bf16, multiply by weight
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    r = y * (1.0 / tl.sqrt(ms + RMS_EPS))
    r = r.to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (r * w).to(tl.bfloat16).to(tl.float32)  # round to bf16 (bf16 * bf16)

    # GELU (exact, erf-based) in fp32
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(out_ptr + row * stride_row + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N, x2.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)
