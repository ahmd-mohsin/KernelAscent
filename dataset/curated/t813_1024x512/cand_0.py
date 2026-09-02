import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 813
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, RMS1_ptr, B2_ptr, G3_ptr, B3_ptr, RMS4_ptr, OUT_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N + offs

    # ---- load matmul output (bf16) ----
    x = tl.load(X_ptr + base)

    # ---- RMSNorm 1 (compute in fp32, round to bf16, then bf16 mult by weight) ----
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    xn = (xf * tl.rsqrt(ms + 1e-6)).to(tl.bfloat16)
    w1 = tl.load(RMS1_ptr + offs)
    x = (xn.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # ---- bias add (bf16) ----
    b2 = tl.load(B2_ptr + offs)
    x = (x.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    # ---- LayerNorm (fp32 accumulation, eps=1e-5), output bf16 ----
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = xf - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.rsqrt(var + 1e-5)
    g3 = tl.load(G3_ptr + offs).to(tl.float32)
    b3 = tl.load(B3_ptr + offs).to(tl.float32)
    xf = diff * inv * g3 + b3
    x = xf.to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    xn = (xf * tl.rsqrt(ms + 1e-6)).to(tl.bfloat16)
    w4 = tl.load(RMS4_ptr + offs)
    x = (xn.to(tl.float32) * w4.to(tl.float32)).to(tl.bfloat16)

    # ---- Softmax (fp32 compute, bf16 output) ----
    yf = x.to(tl.float32)
    mx = tl.max(yf, axis=0)
    e = tl.exp(yf - mx)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT_ptr + base, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = x @ self.W0
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = x + self.b2
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        N = self.W0.shape[1]
        y = torch.matmul(x, self.W0)  # cuBLAS bf16 GEMM (fp32 accumulate)
        y2 = y.reshape(-1, N).contiguous()
        rows = y2.shape[0]
        out = torch.empty_like(y2)

        _fused_post_kernel[(rows,)](
            y2, self.rms1_w, self.b2, self.ln3_g, self.ln3_b, self.rms4_w, out,
            N=N, BLOCK=N, num_warps=4,
        )
        return out.reshape(*orig_shape[:-1], N)
