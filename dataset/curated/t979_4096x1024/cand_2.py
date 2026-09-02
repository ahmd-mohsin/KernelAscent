import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 979
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_epilogue(X, B1, W2, W3, G, Bt, Y,
                    N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)

    # bias add (fp16 rounding, PyTorch half add uses fp32 opmath then casts)
    h = (x + b).to(tl.float16).to(tl.float32)

    # RMSNorm #1
    ms = tl.sum(h * h, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    h = (h * r).to(tl.float16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    h = (h * w2).to(tl.float16).to(tl.float32)

    # RMSNorm #2
    ms = tl.sum(h * h, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    h = (h * r).to(tl.float16).to(tl.float32)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    h = (h * w3).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 stats, eps=1e-5)
    mean = tl.sum(h, axis=0) / N
    d = tl.where(mask, h - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(Bt + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd) * g + bt

    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS fp16 tensor-core GEMM
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue[(m,)](
            h, self.b1, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b, out,
            N=n, BLOCK=BLOCK, num_warps=4,
        )
        return out
