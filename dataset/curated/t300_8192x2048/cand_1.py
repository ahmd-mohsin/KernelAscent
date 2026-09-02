import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_post_kernel(
    X, G, B, W, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * N + offs

    x = tl.load(ptr, mask=mask, other=0.0)  # fp16
    # x * 1.0809 (computed in fp32, rounded to fp16 as PyTorch does)
    v = (x.to(tl.float32) * 1.0809).to(tl.float16)
    # relu
    v = tl.maximum(v, 0.0)
    vf = v.to(tl.float32)

    # layer norm (fp32 internal math, eps=1e-5)
    mean = tl.sum(vf, axis=0) / N
    diff = tl.where(mask, vf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    ln = (diff * inv) * g + b
    ln16 = ln.to(tl.float16)

    # * 1.4484 (fp32 compute, round to fp16)
    h = (ln16.to(tl.float32) * 1.4484).to(tl.float16)

    # rms norm: explicit fp32
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    ms = tl.sum(hf * hf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    normed16 = (hf * r).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    out = (normed16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM with fp32 accumulate
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_post_kernel[(m,)](
            y, self.ln3_g, self.ln3_b, self.rms5_w, out,
            n, BLOCK=BLOCK,
            num_warps=16,
        )
        return out
