import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 239
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_epilogue(X, W3, W4, B, Out, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # load matmul output row (fp16) and upcast for gelu (opmath = fp32)
    x = tl.load(X + row * stride_x + offs).to(tl.float32)

    # exact (erf) GELU, computed in fp32 then cast back to fp16 like PyTorch
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation, result cast to fp16
    mx = tl.max(g, axis=0)
    e = tl.math.exp(g - mx)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm 1: fp32 mean of squares, cast to fp16, fp16 multiply by weight
    ms1 = tl.sum(p * p, axis=0) / BLOCK
    y = (p * tl.math.rsqrt(ms1 + 1e-6)).to(tl.float16)
    w3 = tl.load(W3 + offs)
    y = y * w3

    # RMSNorm 2
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / BLOCK
    y2 = (yf * tl.math.rsqrt(ms2 + 1e-6)).to(tl.float16)
    w4 = tl.load(W4 + offs)
    y2 = y2 * w4

    # bias add (fp16)
    b = tl.load(B + offs)
    out = y2 + b

    tl.store(Out + row * stride_o + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = x @ self.W0

        orig_shape = h.shape
        h2d = h.reshape(-1, orig_shape[-1])
        if not h2d.is_contiguous():
            h2d = h2d.contiguous()
        rows, cols = h2d.shape

        out = torch.empty_like(h2d)
        _fused_epilogue[(rows,)](
            h2d, self.rms3_w, self.rms4_w, self.b5, out,
            h2d.stride(0), out.stride(0),
            BLOCK=cols,
            num_warps=4,
        )
        return out.reshape(orig_shape)
