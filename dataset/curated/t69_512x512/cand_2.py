import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld

SEED = 69
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_rms_softmax_gelu(
    X_ptr, W_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load matmul output (fp16) ----
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, erf-based), computed in fp32, rounded to fp16 like PyTorch ----
    SQRT1_2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tld.erf(x * SQRT1_2))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)

    # ---- RMSNorm (fp32 accumulation, matches reference exactly) ----
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    inv = tld.rsqrt(ms + 1e-6)
    n16 = (gf * inv).to(tl.float16)

    # ---- scale by rms weight (product exact in fp32, rounded to fp16) ----
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v16 = (n16.to(tl.float32) * w).to(tl.float16)
    vf = v16.to(tl.float32)

    # ---- softmax (fp32 accumulation, fp16 output, like PyTorch half softmax) ----
    vf_masked = tl.where(mask, vf, float("-inf"))
    mx = tl.max(vf_masked, axis=0)
    e = tld.exp(vf_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)
    pf = p16.to(tl.float32)

    # ---- final GELU ----
    out = 0.5 * pf * (1.0 + tld.erf(pf * SQRT1_2))
    tl.store(Y_ptr + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)

        if not h.is_cuda:
            # CPU fallback: reference path
            z = F.gelu(h)
            zf = z.float()
            z = (zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(z.dtype) * self.rms2_w
            z = torch.softmax(z, dim=-1)
            return F.gelu(z)

        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_rms_softmax_gelu[(rows,)](
            h, self.rms2_w, y, N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
