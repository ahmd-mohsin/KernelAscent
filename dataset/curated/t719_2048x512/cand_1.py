import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 719
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, g_ptr, b_ptr, w_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- exact GELU (erf-based), round to bf16 like PyTorch output ----
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, bf16 output) ----
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / D
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = d * inv * g + b
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 internally, bf16 output) ----
    zm = tl.where(mask, z, float('-inf'))
    m = tl.max(zm, axis=0)
    e = tl.exp(zm - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm in fp32, cast to bf16, scale by w (fp32 opmath), ReLU ----
    ms = tl.sum(p * p, axis=0) / D
    r = p * tl.math.rsqrt(ms + 1e-6)
    r = r.to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    o = r * w
    o = tl.maximum(o, 0.0)

    tl.store(out_ptr + base + offs, o.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        orig_shape = x.shape
        Dm = orig_shape[-1]
        x2 = x.view(-1, Dm)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dm)
        _fused_kernel[(rows,)](
            x2, self.ln1_g, self.ln1_b, self.rms3_w, out,
            D=Dm, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)
