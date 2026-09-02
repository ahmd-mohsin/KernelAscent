import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 601
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_kernel(X, OUT, G2, B2, G4, B4,
                  N, stride_x, stride_o,
                  EPS: tl.constexpr, SCALE: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X + row * stride_x + cols).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)
    # cast to bf16 and back (op boundary in reference)
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g2 = tl.load(G2 + cols).to(tl.float32)
    b2 = tl.load(B2 + cols).to(tl.float32)
    y = diff * rstd * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax
    mx = tl.max(y, axis=0)
    e = tl.exp(y - mx)
    s = tl.sum(e, axis=0)
    z = e / s
    z = z.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(z, axis=0) / N
    diff2 = z - mean2
    var2 = tl.sum(diff2 * diff2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols).to(tl.float32)
    b4 = tl.load(B4 + cols).to(tl.float32)
    w = diff2 * rstd2 * g4 + b4
    w = w.to(tl.bfloat16).to(tl.float32)

    # scale
    out = (w * SCALE).to(tl.bfloat16)
    tl.store(OUT + row * stride_o + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_kernel[(m,)](
            h, out,
            self.ln2_g, self.ln2_b, self.ln4_g, self.ln4_b,
            n, h.stride(0), out.stride(0),
            EPS=1e-5, SCALE=1.2373,
            BLOCK=512,
            num_warps=4,
        )
        return out
