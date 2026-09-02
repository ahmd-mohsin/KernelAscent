import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 686
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, B1, LN_G, LN_B, W3, W4, B5, Out,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * stride

    # x + b1 (fp16 add == fp32 add then round to fp16)
    x = tl.load(X + base + offs).to(tl.float32)
    b1 = tl.load(B1 + offs).to(tl.float32)
    h = (x + b1).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 accumulate, like PyTorch's half layer_norm)
    mean = tl.sum(h, axis=0) / N
    d = h - mean
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN_G + offs).to(tl.float32)
    b = tl.load(LN_B + offs).to(tl.float32)
    y = (d * inv * g + b).to(tl.float16)

    # RMSNorm 1: fp32 normalize, cast to fp16, multiply by fp16 weight
    yf = y.to(tl.float32)
    r1 = 1.0 / tl.sqrt(tl.sum(yf * yf, axis=0) / N + 1e-6)
    z = (yf * r1).to(tl.float16) * tl.load(W3 + offs)

    # RMSNorm 2
    zf = z.to(tl.float32)
    r2 = 1.0 / tl.sqrt(tl.sum(zf * zf, axis=0) / N + 1e-6)
    o = (zf * r2).to(tl.float16) * tl.load(W4 + offs)

    # + b5 (fp16 add)
    o = o + tl.load(B5 + offs)
    tl.store(Out + base + offs, o)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS half GEMM with fp32 accumulation (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_norm_kernel[(Mrows,)](
            h, self.b1, self.ln2_g, self.ln2_b,
            self.rms3_w, self.rms4_w, self.b5, out,
            N, h.stride(0),
            BLOCK=4096,
            num_warps=16,
            num_stages=1,
        )
        return out
