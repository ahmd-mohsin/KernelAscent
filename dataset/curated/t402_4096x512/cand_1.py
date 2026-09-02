import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 402
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, OUT_ptr,
    LN2G_ptr, LN2B_ptr, RMSW_ptr, LN5G_ptr, LN5B_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # --- exact GELU (erf-based), fp32 math, round to bf16 like PyTorch ---
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.where(mask, g, 0.0)

    # --- LayerNorm 1 (fp32 stats, affine, round to bf16) ---
    mean = tl.sum(g, axis=0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    w2 = tl.load(LN2G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(LN2B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * w2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # --- scale by 1.003 (fp32 math, round to bf16) ---
    y = y * 1.003
    y = y.to(tl.bfloat16).to(tl.float32)
    y = tl.where(mask, y, 0.0)

    # --- RMSNorm (explicit fp32, round to bf16, then * weight, round bf16) ---
    ms = tl.sum(y * y, axis=0) / N
    r = y * (1.0 / tl.sqrt(ms + 1e-6))
    r = r.to(tl.bfloat16).to(tl.float32)
    rw = tl.load(RMSW_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = r * rw
    r = r.to(tl.bfloat16).to(tl.float32)
    r = tl.where(mask, r, 0.0)

    # --- LayerNorm 2 ---
    m2 = tl.sum(r, axis=0) / N
    d2 = tl.where(mask, r - m2, 0.0)
    v2 = tl.sum(d2 * d2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(v2 + 1e-5)
    w5 = tl.load(LN5G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(LN5B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = d2 * inv2 * w5 + b5

    tl.store(OUT_ptr + row * stride + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)

        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 16 if BLOCK >= 4096 else 8
        _fused_post_kernel[(Mrows,)](
            h, out,
            self.ln2_g, self.ln2_b, self.rms4_w, self.ln5_g, self.ln5_b,
            N, h.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out

    def _forward_ref(self, x):
        x = x @ self.W0
        x = F.gelu(x)
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        x = x * 1.003
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        x = F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)
        return x
