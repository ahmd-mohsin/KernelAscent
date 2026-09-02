import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 733
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_norm_gelu_kernel(
    X, W2, G3, B3, W4, Out,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # scale by 1.4626 (fp32 opmath, round to fp16 like PyTorch)
    x = tl.load(X + base + offs).to(tl.float32)
    x = (x * 1.4626).to(tl.float16).to(tl.float32)

    # RMSNorm 2
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)
    w2 = tl.load(W2 + offs).to(tl.float32)
    x = (x * w2).to(tl.float16).to(tl.float32)

    # LayerNorm 3 (fp32 accumulation, eps=1e-5)
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    g = tl.load(G3 + offs).to(tl.float32)
    b = tl.load(B3 + offs).to(tl.float32)
    x = (xc * rstd * g + b).to(tl.float16).to(tl.float32)

    # RMSNorm 4
    ms2 = tl.sum(x * x, axis=0) / N
    x = (x * tl.rsqrt(ms2 + 1e-6)).to(tl.float16).to(tl.float32)
    w4 = tl.load(W4 + offs).to(tl.float32)
    x = (x * w4).to(tl.float16).to(tl.float32)

    # exact GELU (erf-based) in fp32
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(Out + base + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)

        if not h.is_cuda:
            # CPU fallback: reference path
            h = h * 1.4626
            _xf = h.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms2_w
            h = F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)
            _xf = h.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms4_w
            return F.gelu(h)

        h = h.contiguous()
        rows = h.numel() // h.shape[-1]
        N = h.shape[-1]
        out = torch.empty_like(h)
        _fused_norm_gelu_kernel[(rows,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, self.rms4_w, out,
            N=N, BLOCK=N, num_warps=8,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
